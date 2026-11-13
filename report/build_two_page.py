#!/usr/bin/env python3
"""Build the exactly-two-page report: report/two_page.md -> two_page.html -> two_page.pdf.

    uv run --with pypdf report/build_two_page.py            # normal build
    uv run --with pypdf report/build_two_page.py --probe 6  # slack test: adds 6 body lines
                                                             # at the end; writes to a temp dir

Reads two_page.md fresh each run (never writes it). Parses the six figure numbers out of
results/dev_precedent.md and results/test_final2.md into data/two_page_figure.json, and draws
the figure from that file. Makes the PDF with headless Chrome, counts its pages, and exits
non-zero if the count is not exactly 2.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MD = HERE / "two_page.md"
if "--md" in sys.argv:  # experiments only: build another markdown file into a temp dir
    MD = Path(sys.argv[sys.argv.index("--md") + 1]).resolve()
DEV = ROOT / "results" / "dev_precedent.md"
TEST = ROOT / "results" / "test_final2.md"
FIG_JSON = HERE / "data" / "two_page_figure.json"
FOREST_JSON = HERE / "data" / "two_page_forest.json"
CURVES_JSON = HERE / "data" / "two_page_curves.json"
PRED_JSON = HERE / "data" / "two_page_predictions.json"
R = ROOT / "results"
PREDICTIONS_PARQUET = ROOT / "outputs" / "final2" / "predictions.parquet"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MINUS = "−"

# ====================================================================== figure data (parsed, never typed)
SOURCES = [
    ("today", "Today's databases", "today's links"),
    ("targets", "Targets dated to mid 2020", "2020 links"),
    ("names", "Drug names and targets dated to mid 2020", "2020 names and links"),
]
DEV_COL = "gain over registration, lead drugs resampled"
TEST_COL = "AUROC minus logistic: registration"
EST_RE = re.compile(r"([+\-−]\d\.\d+)\s*\(\s*([+\-−]\d\.\d+)\s+to\s+([+\-−]\d\.\d+)\s*\)")


def num(s: str) -> float:
    return float(s.replace(MINUS, "-"))


def md_tables(path: Path):
    """Yield (header_cells, rows) for each pipe table in a markdown file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[i:j]]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            yield rows[0], body
            i = j
        else:
            i += 1


def pull(path: Path, col_name: str, flag_phrase: str):
    rel = str(path.relative_to(ROOT))
    want = f"logistic: registration + precedent ({flag_phrase}, experimental arms only)"
    for head, rows in md_tables(path):
        idx = [k for k, h in enumerate(head) if col_name in h]
        if not idx:
            continue
        for r in rows:
            if r[0] == want:
                m = EST_RE.search(r[idx[0]])
                if not m:
                    raise SystemExit(f"Could not read '{r[idx[0]]}' in {rel} (row '{want}')")
                return {
                    "estimate": num(m.group(1)), "lo": num(m.group(2)), "hi": num(m.group(3)),
                    "source": f"{rel}: row '{r[0]}', column '{head[idx[0]]}'",
                }
    raise SystemExit(f"Row '{want}' with column '{col_name}' not found in {rel}")


def figure_data() -> dict:
    rows = []
    for key, label, phrase in SOURCES:
        rows.append({
            "id": key, "label": label,
            "development": pull(DEV, DEV_COL, phrase),
            "test": pull(TEST, TEST_COL, phrase),
        })
    data = {
        "what": "Gain in AUROC over the registration baseline from adding the precedent flag "
                "(strict: experimental arms only), 95% intervals.",
        "rows": rows,
    }
    FIG_JSON.parent.mkdir(exist_ok=True)
    FIG_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json.loads(FIG_JSON.read_text(encoding="utf-8"))  # the chart reads the file, not memory


# ====================================================================== forest data (parsed, never typed)
def md_tables_h(path: Path):
    """Like md_tables, but also yields the nearest heading above each table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i, h2, h3 = 0, "", ""
    while i < len(lines):
        if lines[i].startswith("###"):
            h3 = lines[i].lstrip("# ").strip()
        elif lines[i].startswith("#"):
            h2, h3 = lines[i].lstrip("# ").strip(), ""
        heading = f"{h2} > {h3}" if h3 else h2
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[i:j]]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            yield rows[0], body, heading
            i = j
        else:
            i += 1


def find_est(path: Path, name: str, col_sub: str, heading_sub: str = "", name_col: int = 0) -> dict:
    """First row called `name` in a table whose column header contains `col_sub`; parse 'est (lo to hi)'."""
    rel = str(path.relative_to(ROOT))
    for head, rows, heading in md_tables_h(path):
        if heading_sub and heading_sub not in heading:
            continue
        idx = [k for k, h in enumerate(head) if col_sub in h]
        if not idx:
            continue
        for r in rows:
            if r[name_col] == name:
                m = EST_RE.search(r[idx[0]])
                if m:
                    return {"estimate": num(m.group(1)), "lo": num(m.group(2)), "hi": num(m.group(3)),
                            "source": f"{rel}: row '{name}', column '{head[idx[0]]}'"
                                      + (f", section '{heading}'" if heading_sub else "")}
    return {}


def find_auroc(path: Path, name: str) -> float | None:
    for head, rows, _ in md_tables_h(path):
        if head and head[0].lower().startswith("model") and any("AUROC (95%" in h for h in head):
            for r in rows:
                if r[0] == name:
                    return float(re.match(r"[\d.]+", r[1]).group(0))
    return None


def find_hier(variant: str, links: str) -> dict:
    path = R / "dev_hierarchy.md"
    rel = str(path.relative_to(ROOT))
    for head, rows, _ in md_tables_h(path):
        if head[:2] == ["variant", "links"] and "gain" in head:
            for r in rows:
                if r[0] == variant and r[1] == links:
                    g = head.index("gain"); iv = [k for k, h in enumerate(head) if "drugs resampled" in h][0]
                    m = re.match(r"\s*([+\-−]\d\.\d+)\s+to\s+([+\-−]\d\.\d+)", r[iv])
                    return {"estimate": num(r[g]), "lo": num(m.group(1)), "hi": num(m.group(2)),
                            "source": f"{rel}: row '{variant}' / links '{links}', columns 'gain' and '{head[iv]}'"}
    return {}


# (group, plain label, test model name, test file, dev finder, leaky)
T1 = R / "test_final2_item7_final.md"      # the first run's models, lead-drug intervals
T2 = R / "test_final2.md"                  # the second run
T1COL, T2COL = "AUROC minus registration", "AUROC minus logistic: registration"
DEVM, DEVP, DEVW = R / "dev_models.md", R / "dev_precedent.md", R / "dev_whathow.md"
PREC_T = "logistic: registration + precedent ({}, experimental arms only)"
FOREST_ROWS = [
    ("First-run models", "Phase alone", "lookup: phase", T1, "m", False),
    ("First-run models", "Adds words and three more tables",
     "ladder: registration + endpoint + history + genetics + words", T1, "m", False),
    ("First-run models", "Boosted trees on the extra tables",
     "ladder: boosted trees, endpoint + history + genetics", T1, "m", False),
    ("First-run models", "Adds all public biology sources", "biology: registration + all sources", T1, "m", False),
    ("First-run models", "Leaky: fields edited after the trial", "leaky: registration + post-hoc fields", T1, "m", True),
    ("Hierarchical models", "Disease, drug, target, sponsor effects", "hierarchical: full hierarchy", T2, ("h", "full hierarchy", "today"), False),
    ("Hierarchical models", "The same, plus precedent and genetics", "hierarchical: full + precedent + genetics", T2,
     ("h", "full + precedent + genetics", "today"), False),
    ("Precedent: same target already in Phase 3", "Today's databases", PREC_T.format("today's links"), T2,
     ("p", PREC_T.format("today's links")), False),
    ("Precedent: same target already in Phase 3", "Drug names and links dated to mid 2020", PREC_T.format("2020 names and links"), T2,
     ("p", PREC_T.format("2020 names and links")), False),
    ("Disease and target, dated to mid 2020", "Disease, target and biology only", "logistic: WHAT (2020 names and links)", T2, None, False),
    ("Disease and target, dated to mid 2020", "Registration + disease, target, biology",
     "logistic: HOW + WHAT (2020 names and links)", T2, ("w",), False),
]


def forest_data() -> dict:
    rows, missing = [], []
    for group, label, model, tfile, dev, leaky in FOREST_ROWS:
        t = find_est(tfile, model, T1COL if tfile == T1 else T2COL)
        if not t:
            missing.append(f"{label} ({model}): not found in {tfile.name}")
            continue
        d = {}
        if dev == "m":
            d = find_est(DEVM, model, "AUROC minus registration", "Time-ordered")
        elif dev and dev[0] == "h":
            d = find_hier(dev[1], dev[2])
        elif dev and dev[0] == "p":
            d = find_est(DEVP, dev[1], "gain over registration, lead drugs resampled")
        elif dev and dev[0] == "w":
            d = find_est(DEVW, "HOW + WHAT minus the registration baseline", "AUROC difference", "2020 names and links")
        rows.append({"group": group, "label": label, "model": model, "leaky": leaky, "test": t,
                     "development": d or None})
    base = find_auroc(T1, "logistic: registration")
    data = {
        "what": "AUROC minus the registration baseline's, test set (1,236 trials finished 2021 to 2023), 95% intervals "
                "resampling lead drugs; development-validation difference beside it where a committed dev file has one.",
        "baseline_auroc": base, "rows": rows, "left_out": missing,
    }
    FOREST_JSON.parent.mkdir(exist_ok=True)
    FOREST_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json.loads(FOREST_JSON.read_text(encoding="utf-8"))


# ====================================================================== curve data (from the committed predictions file)
CURVE_MODELS = [  # (column in predictions file, short label for the key)
    ("logistic: registration", "Registration baseline"),
    ("lookup: phase", "Phase alone"),
    ("hierarchical: full hierarchy", "Hierarchical: disease, target, sponsor"),
    ("logistic: WHAT (2020 names and links)", "Disease, target and biology only"),
    ("logistic: HOW + WHAT (2020 names and links)", "Registration + disease, target, biology"),
]


def _curves(scores, y):
    """ROC and precision-recall points (ties collapsed) plus AUROC and average precision. Pure python."""
    order = sorted(range(len(y)), key=lambda i: -scores[i])
    P = sum(y); N = len(y) - P
    tp = fp = 0
    roc, pr = [(0.0, 0.0)], []
    k = 0
    while k < len(order):
        j = k
        while j < len(order) and scores[order[j]] == scores[order[k]]:
            tp += y[order[j]]; fp += 1 - y[order[j]]; j += 1
        roc.append((fp / N, tp / P)); pr.append((tp / P, tp / (tp + fp)))
        k = j
    auc = sum((roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i - 1][1]) / 2 for i in range(1, len(roc)))
    ap, prev = 0.0, 0.0
    for r, p in pr:
        ap += (r - prev) * p; prev = r
    pr = [(0.0, pr[0][1])] + pr
    return roc, pr, auc, ap


def _thin(pts, n=150):
    if len(pts) <= n:
        return [[round(a, 4), round(b, 4)] for a, b in pts]
    idx = sorted({round(i * (len(pts) - 1) / (n - 1)) for i in range(n)})
    return [[round(pts[i][0], 4), round(pts[i][1], 4)] for i in idx]


def curves_data() -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        if CURVES_JSON.exists():
            print("note: pyarrow not available; using the saved data/two_page_curves.json", file=sys.stderr)
            return json.loads(CURVES_JSON.read_text(encoding="utf-8"))
        raise SystemExit("pyarrow is needed once to draw the curves: uv run --with pypdf --with pyarrow report/build_two_page.py")
    if not PREDICTIONS_PARQUET.exists() and CURVES_JSON.exists():
        print("note: no outputs/final2/predictions.parquet; using the saved data/two_page_curves.json", file=sys.stderr)
        return json.loads(CURVES_JSON.read_text(encoding="utf-8"))
    names = pq.ParquetFile(PREDICTIONS_PARQUET).schema.names
    out, left = [], []
    wanted = [c for c, _ in CURVE_MODELS if c in names]
    tbl = pq.read_table(PREDICTIONS_PARQUET, columns=["outcome"] + wanted).to_pydict()
    for col, short in CURVE_MODELS:
        if col not in names:
            left.append({"column": col, "label": short, "reason": "not a column in outputs/final2/predictions.parquet"})
            continue
        pairs = [(float(s), int(o)) for s, o in zip(tbl[col], tbl["outcome"]) if s is not None and s == s and o is not None]
        roc, pr, auc, ap = _curves([a for a, _ in pairs], [b for _, b in pairs])
        out.append({"column": col, "label": short, "n": len(pairs),
                    "roc": {"points": _thin(roc), "area": round(auc, 3)},
                    "pr": {"points": _thin(pr), "area": round(ap, 3)}})
    ys = tbl["outcome"]
    data = {
        "what": "Test-set ROC and precision-recall curves; points thinned to at most 150 per curve; areas are AUROC and "
                "average precision (step-wise area under the precision-recall curve).",
        "source": "outputs/final2/predictions.parquet (written by the committed test run, TEST_PLAN_2.md item 6): "
                  "columns 'outcome' and one model column each; nothing else read",
        "n": len(ys), "base_rate": round(sum(ys) / len(ys), 4), "curves": out, "left_out": left,
    }
    CURVES_JSON.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return json.loads(CURVES_JSON.read_text(encoding="utf-8"))


# ====================================================================== predictions table (parsed from the scorecard)
SCORE = R / "test_final2_item7_scorecard.md"
SCORE_FALLBACK = R / "test_scorecard.md"
PRED_LABELS = {  # id -> plain-words name of the prediction (the numbers all come from the scorecard)
    "1a": "Trials that succeed",
    "1b": "Baseline over-predicts success",
    "2a": "Baseline AUROC",
    "2b": "Registration fields that matter most",
    "3": "AUROC gain from leaked fields",
    "4a": "Biologics against small molecules",
    "4b": "AUROC gain from knowing drug type",
    "5a": "Active control against placebo",
    "5b": "Under 100 against 300+ participants",
    "6a": "Success with genetic support",
    "6b": "Same, drugs with 5 or fewer earlier trials",
    "7a": "Rare-disease recruitment stops, ratio, non-cancer",
    "7b": "Success in rare disease",
}
PRED_UNITS = {}


def _plain(cell: str, strip_zero: bool) -> str:
    """Rates and point differences become 'per 100'; minus signs become real minus signs."""
    t = cell.replace("%", "").replace(" points", "").replace("-", MINUS)
    if strip_zero:
        t = re.sub(r"(\d)\.0\b", r"\1", t)
    return t


def predictions_data() -> dict:
    path = SCORE
    rel = str(path.relative_to(ROOT))
    text = path.read_text(encoding="utf-8")
    rows = {}
    for head, body, _ in md_tables_h(path):
        if head[:2] == ["#", "Prediction"]:
            for r in body:
                rows[r[0]] = dict(zip(head, r))
    fallback = {}
    if SCORE_FALLBACK.exists():
        for head, body, _ in md_tables_h(SCORE_FALLBACK):
            if head[:2] == ["#", "Prediction"]:
                for r in body:
                    fallback[r[0]] = dict(zip(head, r))
    out = []
    order = [k for k in PRED_LABELS if k != "2b"]
    for pid in PRED_LABELS:
        if pid == "2b":
            m = re.search(r"\*\*2b\. Predicted: (.+?)\. Observed: (.+?)\. (Matches|Does not match|Differs)", text)
            if not m:
                raise SystemExit(f"Prediction 2b not found in {rel}")
            clean = lambda x: x.replace("sponsor_class", "sponsor class").replace("_", " ").replace(" and ", ", ")
            ok = m.group(3) == "Matches"
            out.append({"id": "2b", "label": PRED_LABELS["2b"], "expected": re.sub(r" matter most$", "", clean(m.group(1))).capitalize(),
                        "observed": clean(m.group(2)).capitalize() + (" (same)" if ok else " (differs)"),
                        "mark": "in" if ok else "out", "verdict": "matches" if ok else "does not match",
                        "source": f"{rel}: line '2b. Predicted ... Observed ...'"})
            continue
        r = rows.get(pid) or fallback.get(pid)
        if not r:
            raise SystemExit(f"Prediction {pid} not found in {rel} or {SCORE_FALLBACK.name}")
        for key in ("Predicted", "Observed (95% interval)", "Verdict"):
            if not r.get(key) and fallback.get(pid, {}).get(key):
                r[key] = fallback[pid][key]
        v = r["Verdict"]
        mark = ("in" if v.startswith("inside") else "near" if v.startswith("outside, but") else
                "out" if v.startswith("outside") else "none")
        exp = r["Predicted"]
        if exp.startswith("no range"):
            m = re.search(r"like (\d+)% vs (\d+)%", r["Prediction"])
            n = re.search(r"email: about ([\d.]+)", r.get("Notes", ""))
            exp = (f"{m.group(1)} against {m.group(2)} per 100" if m else
                   f"about {n.group(1)}" if n else "weaker (no figure)")
        else:
            exp = _plain(exp, True)
        per = "%" in r["Observed (95% interval)"] or "points" in r["Observed (95% interval)"]
        obs = _plain(r["Observed (95% interval)"], False)
        out.append({"id": pid, "label": PRED_LABELS[pid] + (", per 100" if per else ""), "expected": exp, "observed": obs, "mark": mark,
                    "verdict": v, "source": f"{rel}: row {pid}"})
    # second-round line, from test_final2.md
    t2 = T2.read_text(encoding="utf-8")
    met, notmet, rule, item = 0, [], "", ""
    for ln in t2.splitlines():
        mi = re.match(r"## Item (\d+)", ln)
        if mi:
            item = mi.group(1)
        if "**Verdict:**" in ln:
            sub = re.match(r"\*\*\(([a-z])\)", ln)
            seg = ln.split("**Verdict:**", 1)[1].split("against the stated range")[0]
            rm = re.search(r"plan's rule \(([^)]+)\)", seg)
            rule = rule or (rm.group(1) if rm else "")
            toks = re.findall(r"\*\*(not met|met)\*\*", seg)
            for k, tk in enumerate(toks):
                tag = item + (sub.group(1) if sub else "")
                if len(toks) > 1:
                    tag += " (" + ("first part" if k == 0 else "second part") + ")"
                if tk == "met":
                    met += 1
                else:
                    notmet.append(tag)
    total = met + len(notmet)
    data = {"rows": out, "second_round": {"met": met, "total": total, "not_met": notmet, "rule": rule,
                                          "source": str(T2.relative_to(ROOT)) + ": every 'Verdict' line"}}
    PRED_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json.loads(PRED_JSON.read_text(encoding="utf-8"))


# ====================================================================== inline markdown
def inline(s: str) -> str:
    stash: list[str] = []

    def keep(m):
        stash.append('<mark class="john">' + html.escape(m.group(0), quote=False) + "</mark>")
        return f"\x00{len(stash) - 1}\x00"

    s = re.sub(r"\[\[JOHN:.*?\]\]", keep, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)


# ====================================================================== block markdown
def parse_blocks(text: str):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    blocks, para, i = [], [], 0

    def flush():
        if para:
            blocks.append(("p", " ".join(x.strip() for x in para)))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if not st:
            flush(); i += 1; continue
        m = re.match(r"(#{1,4})\s+(.*)", st)
        if m:
            flush(); blocks.append((f"h{len(m.group(1))}", m.group(2))); i += 1; continue
        m = re.fullmatch(r"\{\{\s*table:\s*([\w-]+)\s*\}\}", st)
        if m:
            flush(); blocks.append(("tablemark", m.group(1))); i += 1; continue
        if re.fullmatch(r"\{\{\s*break\s*\}\}", st):
            flush(); blocks.append(("break", "")); i += 1; continue
        m = re.fullmatch(r"\{\{\s*figure:\s*([\w-]+)\s*\}\}", st)
        if m:
            flush(); blocks.append(("figure", m.group(1))); i += 1; continue
        if st.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[i:j]]
            aligns = None
            if len(rows) > 1 and all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
                aligns = ["right" if c.endswith(":") and not c.startswith(":") else
                          "center" if c.startswith(":") and c.endswith(":") else "left" for c in rows[1]]
                rows.pop(1)
            blocks.append(("table", (rows[0], rows[1:], aligns))); i = j; continue
        if re.match(r"[-*•]\s+", st) or re.match(r"\d+[.)]\s+", st):
            flush()
            ordered = bool(re.match(r"\d+[.)]\s+", st))
            items = []
            while i < len(lines):
                s2 = lines[i].strip()
                mm = re.match(r"(?:[-*•]|\d+[.)])\s+(.*)", s2)
                if mm and (re.match(r"\d+[.)]\s", s2) is not None) == ordered:
                    items.append(mm.group(1)); i += 1
                elif s2 and items and lines[i].startswith((" ", "\t")):
                    items[-1] += " " + s2; i += 1
                else:
                    break
            blocks.append(("ol" if ordered else "ul", items)); continue
        para.append(ln); i += 1
    flush()
    return blocks


# ====================================================================== figure drawing
def fnum(x: float, signed=True) -> str:
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", MINUS)


def figure_svg(data: dict) -> str:
    rows = data["rows"]
    W = 254.0
    x_tag, x0, x1, x_val = 2.0, 70.0, 212.0, 252.0
    lo_all = min(min(r["development"]["lo"], r["test"]["lo"]) for r in rows)
    hi_all = max(max(r["development"]["hi"], r["test"]["hi"]) for r in rows)
    a = -0.005 * (int(-lo_all / 0.005) + 1) if lo_all < 0 else 0.0
    b = 0.005 * (int(hi_all / 0.005) + 1)
    X = lambda v: x0 + (v - a) / (b - a) * (x1 - x0)
    top, gh, rowh = 6.0, 40.0, 13.0
    H = top + gh * len(rows) + 26
    out = []
    ticks, t = [], round(-0.01 * int(-a / 0.01 + 1e-9), 3) if a < 0 else 0.0
    while t <= b + 1e-9:
        ticks.append(round(t, 3)); t += 0.01
    ybot = top + gh * len(rows) - 6
    for t in ticks:
        out.append(f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{X(t):.1f}" y1="{top}" x2="{X(t):.1f}" y2="{ybot}"/>')
        txt = "0" if abs(t) < 1e-9 else fnum(t)
        out.append(f'<text class="tick{" zerot" if abs(t) < 1e-9 else ""}" x="{X(t):.1f}" y="{ybot + 10}" text-anchor="middle">{txt}</text>')
    out.append(f'<text class="axt" x="{(x0 + x1) / 2:.1f}" y="{ybot + 21}" text-anchor="middle">gain in AUROC over registration baseline</text>')
    for k, r in enumerate(rows):
        y = top + k * gh
        out.append(f'<text class="grp" x="{x_tag}" y="{y + 7:.1f}">{html.escape(r["label"])}</text>')
        for j, (key, name, cls) in enumerate((("development", "Training set", "dev"), ("test", "Test set", "test"))):
            d = r[key]
            cy = y + 17 + j * rowh
            title = (f'{r["label"]}, {name.lower()}: {fnum(d["estimate"])} '
                     f'({fnum(d["lo"])} to {fnum(d["hi"])})')
            out.append(f'<g><title>{html.escape(title)}</title>')
            out.append(f'<text class="tag {cls}-t" x="{x_tag + 4}" y="{cy + 2.8:.1f}">{name}</text>')
            out.append(f'<line class="ci {cls}-s" x1="{X(d["lo"]):.1f}" y1="{cy}" x2="{X(d["hi"]):.1f}" y2="{cy}"/>')
            for e in (d["lo"], d["hi"]):
                out.append(f'<line class="ci {cls}-s" x1="{X(e):.1f}" y1="{cy - 2.6}" x2="{X(e):.1f}" y2="{cy + 2.6}"/>')
            cx = X(d["estimate"])
            if cls == "dev":
                out.append(f'<circle class="mk dev-f" cx="{cx:.1f}" cy="{cy}" r="3.4"/>')
            else:
                out.append(f'<path class="mk test-f" d="M{cx:.1f} {cy - 4.3} L{cx + 4.3:.1f} {cy} L{cx:.1f} {cy + 4.3} L{cx - 4.3:.1f} {cy} Z"/>')
            out.append(f'<text class="val {cls}-t" x="{x_val}" y="{cy + 2.8:.1f}" text-anchor="end">{fnum(d["estimate"])}</text></g>')
    return (f'<svg class="chart" viewBox="0 0 {W:g} {H:g}" role="img" '
            f'aria-label="Dot and interval chart of the AUROC gain from the precedent flag, by how the sources are dated, '
            f'training set and test set">' + "".join(out) + "</svg>")


def figure_html(data: dict) -> str:
    return (
        '<figure class="fig"><div class="figchart">' + figure_svg(data) + '</div>'
        '<figcaption><div class="take">The more carefully the sources are dated, the less precedent adds</div>'
        '<p class="cap">Gain in AUROC over the registration baseline, from one added fact: another drug on the same target '
        'had already reached Phase 3 in the same disease. AUROC is the chance that a random success is ranked above a random '
        'failure. Bars are 95% intervals. Training set: trials finished before 2021. Test: trials finished 2021 to 2023.</p>'
        '</figcaption></figure>'
    )


# ====================================================================== new figures: forest, curves; predictions table
def forest_svg(data: dict) -> str:
    rows = data["rows"]
    W = 268.0
    xl, x0, x1, xv = 2.0, 142.0, 234.0, 266.0
    a, b = -0.05, 0.07
    X = lambda v: x0 + (min(max(v, a), b) - a) / (b - a) * (x1 - x0)
    pitch, gh, top = 12.6, 9.6, 1.0
    out, y = [], top
    ngroups = len(dict.fromkeys(r["group"] for r in rows))
    ybot = top + gh * ngroups + pitch * len(rows) + 1
    ticks = [-0.04, 0.0, 0.04]
    for t in ticks:
        z = abs(t) < 1e-9
        out.append(f'<line class="{"zero" if z else "grid"}" x1="{X(t):.1f}" y1="{top}" x2="{X(t):.1f}" y2="{ybot:.1f}"/>')
        out.append(f'<text class="tick{" zerot" if z else ""}" x="{X(t):.1f}" y="{ybot + 8.5:.1f}" text-anchor="middle">{"0" if z else fnum(t)}</text>')
    out.append(f'<text class="axt" x="{xv}" y="{ybot + 17.5:.1f}" text-anchor="end">AUROC minus the baseline\'s (right is better)</text>')
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            out.append(f'<text class="grp" x="{xl}" y="{y + 7.2:.1f}">{html.escape(group)}</text>')
            y += gh
        t, d, leaky = r["test"], r["development"], r["leaky"]
        cyt, cyd = y + 3.9, y + 9.3
        tip = f'{r["label"]}. Test set: {fnum(t["estimate"])} ({fnum(t["lo"])} to {fnum(t["hi"])})'
        if d:
            tip += f'. Training-set validation: {fnum(d["estimate"])} ({fnum(d["lo"])} to {fnum(d["hi"])})'
        out.append(f'<g><title>{html.escape(tip)}</title>')
        out.append(f'<text class="rowl{" leakyl" if leaky else ""}" x="{xl}" y="{y + 8.2:.1f}">{html.escape(r["label"])}</text>')

        def mark(v, cy, cls, size, sw):
            lo_off, hi_off, est_off = v["lo"] < a, v["hi"] > b, v["estimate"] < a
            xa, xb = X(v["lo"]), X(v["hi"])
            o = [f'<line class="ci {cls}-s" style="stroke-width:{sw}" x1="{xa:.1f}" y1="{cy}" x2="{xb:.1f}" y2="{cy}"/>']
            for e, off in ((v["lo"], lo_off), (v["hi"], hi_off)):
                if not off:
                    o.append(f'<line class="ci {cls}-s" style="stroke-width:{sw}" x1="{X(e):.1f}" y1="{cy - size * .7:.1f}" x2="{X(e):.1f}" y2="{cy + size * .7:.1f}"/>')
            cx = X(v["estimate"])
            if est_off:
                o.append(f'<path class="mk {cls}-f" d="M{x0 - 2:.1f} {cy} L{x0 + size * 1.8:.1f} {cy - size * 1.1:.1f} L{x0 + size * 1.8:.1f} {cy + size * 1.1:.1f} Z"/>')
            elif cls == "dev":
                o.append(f'<circle class="mk dev-f" cx="{cx:.1f}" cy="{cy}" r="{size}"/>')
            else:
                dd = f"M{cx:.1f} {cy - size:.1f} L{cx + size:.1f} {cy} L{cx:.1f} {cy + size:.1f} L{cx - size:.1f} {cy} Z"
                o.append(f'<path class="mk {"test-h" if leaky else "test-f"}" d="{dd}"/>')
            return "".join(o)

        if d:
            out.append(mark(d, cyd, "dev", 1.9, 0.8))
        out.append(mark(t, cyt, "test", 3.3, 1.4))
        out.append(f'<text class="val test-t" x="{xv}" y="{cyt + 2.7:.1f}" text-anchor="end">{fnum(t["estimate"])}</text></g>')
        y += pitch
    ly = ybot + 27.5
    out.append(f'<path class="mk test-f" d="M5 {ly} l3.3 3.3 l-3.3 3.3 l-3.3 -3.3 Z"/><text class="axs" x="11" y="{ly + 5.6:.1f}">test set</text>')
    out.append(f'<path class="mk test-h" d="M52 {ly} l3.3 3.3 l-3.3 3.3 l-3.3 -3.3 Z"/><text class="axs" x="58" y="{ly + 5.6:.1f}">hollow = leaky</text>')
    out.append(f'<circle class="mk dev-f" cx="119" cy="{ly + 3.3:.1f}" r="1.9"/><text class="axs" x="124" y="{ly + 5.6:.1f}">training set</text>')
    out.append(f'<path class="mk grey-f" d="M172 {ly + 3.3:.1f} l5 -3 v6 Z"/><text class="axs" x="180" y="{ly + 5.6:.1f}">off the scale</text>')
    H = ly + 10
    return (f'<svg class="chart" viewBox="0 0 {W:g} {H:.1f}" role="img" aria-label="Forest plot: AUROC difference from the registration '
            f'baseline on the test set, with 95% intervals, for eleven models, and the training-set validation difference beside it where one exists">'
            + "".join(out) + "</svg>")


CURVE_STYLES = [  # colour and line style together; the baseline is the thick solid orange line
    ("var(--test)", 2.0, ""),
    ("var(--c2)", 1.25, "5 2.4"),
    ("var(--c3)", 1.25, "1.2 2.2"),
    ("var(--c4)", 1.25, "6 2 1.4 2"),
    ("var(--ink2)", 1.25, "2.5 1.5"),
]


def curves_svg(data: dict) -> str:
    cv = data["curves"]
    W, pw, ph = 244.0, 92.0, 104.0
    top = 11.0
    panels = [(24.0, "roc", "ROC curve"), (146.0, "pr", "Precision-recall curve")]
    out = []
    for px, kind, title in panels:
        xs = lambda v, px=px: px + v * pw
        ys = lambda v: top + (1 - v) * ph
        out.append(f'<text class="ptitle" x="{px - 22:.1f}" y="7.5">{title}</text>')
        for t in (0, 0.5, 1):
            out.append(f'<line class="grid" x1="{xs(t):.1f}" y1="{top}" x2="{xs(t):.1f}" y2="{top + ph}"/>'
                       f'<line class="grid" x1="{px}" y1="{ys(t):.1f}" x2="{px + pw}" y2="{ys(t):.1f}"/>')
            out.append(f'<text class="tick" x="{xs(t):.1f}" y="{top + ph + 8.5}" text-anchor="middle">{"0" if t == 0 else "1" if t == 1 else ".5"}</text>')
            out.append(f'<text class="tick" x="{px - 3}" y="{ys(t) + 2.6:.1f}" text-anchor="end">{"0" if t == 0 else "1" if t == 1 else ".5"}</text>')
        out.append(f'<rect class="frame" x="{px}" y="{top}" width="{pw}" height="{ph}"/>')
        if kind == "roc":
            out.append(f'<line class="diag" x1="{xs(0)}" y1="{ys(0)}" x2="{xs(1)}" y2="{ys(1)}"/>')
        else:
            br = data["base_rate"]
            out.append(f'<line class="diag" x1="{xs(0)}" y1="{ys(br):.1f}" x2="{xs(1)}" y2="{ys(br):.1f}"/>')
            out.append(f'<text class="axs" x="{xs(1) - 1:.1f}" y="{ys(br) + 8:.1f}" text-anchor="end">success rate {br:.3f}</text>')
        for k, c in reversed(list(enumerate(cv))):
            col, w, dash = CURVE_STYLES[k]
            pts = c[kind]["points"]
            d = "M" + " L".join(f"{xs(x):.1f} {ys(y):.1f}" for x, y in pts)
            out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}"'
                       + (f' stroke-dasharray="{dash}"' if dash else "") + ' stroke-linejoin="round"/>')
        out.append(f'<text class="axt" x="{px + pw / 2:.1f}" y="{top + ph + 17.5}" text-anchor="middle">'
                   + ("false positive rate" if kind == "roc" else "recall") + "</text>")
        yl = "true positive rate" if kind == "roc" else "precision"
        out.append(f'<text class="axt" transform="translate({px - 15.5:.1f} {top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">{yl}</text>')
    ky = top + ph + 24
    out.append(f'<text class="axs" x="{W - 40}" y="{ky + 6}" text-anchor="end">ROC area</text>')
    out.append(f'<text class="axs" x="{W}" y="{ky + 6}" text-anchor="end">PR area</text>')
    ky += 8.5
    for k, c in enumerate(sorted(cv, key=lambda c: -c["roc"]["area"])):
        k0 = [q["column"] for q in cv].index(c["column"])
        col, w, dash = CURVE_STYLES[k0]
        yy = ky + k * 9.4
        out.append(f'<line x1="1" y1="{yy + 4.2:.1f}" x2="19" y2="{yy + 4.2:.1f}" stroke="{col}" stroke-width="{w}"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
        out.append(f'<text class="rowl{" bold" if k0 == 0 else ""}" x="23" y="{yy + 7:.1f}">{html.escape(c["label"])}</text>')
        out.append(f'<text class="val" x="{W - 40}" y="{yy + 7:.1f}" text-anchor="end">{c["roc"]["area"]:.3f}</text>')
        out.append(f'<text class="val" x="{W}" y="{yy + 7:.1f}" text-anchor="end">{c["pr"]["area"]:.3f}</text>')
    H = ky + len(cv) * 9.4 + 1
    return (f'<svg class="chart" viewBox="0 0 {W:g} {H:.1f}" role="img" aria-label="ROC and precision-recall curves on the test set '
            f'for the registration baseline and three other models, with the area under each">' + "".join(out) + "</svg>")


def forest_cell(data: dict) -> str:
    return ('<figure class="fig cell"><div class="take">Honest models gain little over the registration baseline; hindsight gains most</div>'
            '<div class="figchart">' + forest_svg(data) + '</div>'
            f'<p class="cap">Test-set AUROC minus the registration baseline\'s ({data["baseline_auroc"]:.3f}), with 95% intervals. '
            'AUROC is the chance a random success is ranked above a random failure. Test set: 1,236 trials finished '
            '2021 to 2023; training set: before 2021.</p></figure>')


def curves_cell(data: dict) -> str:
    return ('<figure class="fig cell"><div class="take">The better models sit only a little above the baseline curve</div>'
            '<div class="figchart">' + curves_svg(data) + '</div>'
            '<p class="cap">Left: the share of successes ranked above a cut-off, against the share of failures also ranked above it. '
            'Right: of the trials above the cut-off, the share that succeeded, against the share of all successes found. '
            'Dotted grey: guessing. Areas: AUROC, and average precision. Test set: 1,236 trials finished 2021 to 2023.</p></figure>')


MARKS = {"in": ("vm in", "inside the predicted range"),
         "near": ("vm near", "outside the range, but the interval reaches it"),
         "out": ("vm out", "outside the range, and so is the whole interval"),
         "none": ("vm none", "no range was stated")}


def predictions_html(data: dict) -> str:
    rows = data["rows"]
    body = "".join(
        f'<tr><td class="pid">{html.escape(r["id"])}</td><td>{html.escape(r["label"])}</td><td>{html.escape(r["expected"])}</td>'
        f'<td>{html.escape(r["observed"])}</td><td class="vc"><span class="{MARKS[r["mark"]][0]}" role="img" '
        f'aria-label="{html.escape(MARKS[r["mark"]][1])}"></span></td></tr>' for r in rows)
    used = [m for m in ("in", "near", "out", "none") if any(r["mark"] == m for r in rows)]
    key = "; ".join(f'<span class="{MARKS[m][0]}"></span> {MARKS[m][1]}' for m in used)
    sr = data["second_round"]
    line = (f'<p class="predline"><strong>Second round.</strong> By its own rule ({html.escape(sr["rule"])}), '
            f'{sr["met"]} of the plan\'s {sr["total"]} expectations were met; not met: {html.escape(", ".join(sr["not_met"]))}.</p>')
    return ('<div class="predwrap"><table class="pred"><colgroup><col class="c0"><col class="c1"><col class="c2"><col class="c3"><col class="c4"></colgroup><thead><tr><th></th><th>Prediction</th><th>Expected</th>'
            '<th>Observed (95% interval)</th><th></th></tr></thead><tbody>' + body + '</tbody></table>'
            f'<p class="key">Mark: {key}.</p>'
            + '</div>')  # the second-round line is left out of the report; `line` is kept for reference


# ====================================================================== page
LIGHT = """
  --bg:#f3f5f7; --surface:#ffffff; --ink:#121b24; --ink2:#3b4753; --muted:#62707d; --rule:#d6dce2; --wash:#eaeef2;
  --dev:#2a78d6; --dev-ink:#1b5aa8; --test:#eb6834; --test-ink:#a63d0e; --c2:#121b24; --c3:#7a4fb0; --c4:#1f8a70;
  --john-bg:#e7dffb; --john-ink:#38287a;
"""
DARK = """
  --bg:#0e141a; --surface:#151d25; --ink:#e9eef3; --ink2:#c3ccd5; --muted:#98a6b3; --rule:#2b3743; --wash:#1d2731;
  --dev:#3987e5; --dev-ink:#8db9f2; --test:#d95926; --test-ink:#f2a07c; --c2:#e9eef3; --c3:#b79af0; --c4:#4fc2a0;
  --john-bg:#2b2447; --john-ink:#cfc3f7;
"""

CSS = """
/* Layout concept: the long report's lab-report page, cut down to A4 x 2. Title band, then two ruled text columns;
   blue = development data, orange = test set; claims as headings; one small dot-and-interval figure in a column. */
:root { @LIGHT@
  --font:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --head:'IBM Plex Sans Condensed','IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { @DARK@ color-scheme: dark; } }
:root[data-theme="dark"] { @DARK@ color-scheme: dark; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 var(--font); }
.page { max-width: 760px; margin: 0 auto; padding-block: 28px 56px;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }
h1, h2, h3 { font-family: var(--head); text-wrap: balance; line-height:1.2; margin:0; }
h1 { font-size: clamp(28px, 6vw, 38px); font-weight:600; letter-spacing:-0.01em; }
h2 { font-size: 22px; font-weight:600; margin: 32px 0 10px; }
h3 { font-size: 17px; font-weight:600; margin: 22px 0 8px; }
.hero { border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 4px; }
.byline { color: var(--ink2); margin: 6px 0 0; }
p { margin: 0 0 12px; }
.lede { font-size: 1.12em; line-height: 1.5; }
ul, ol { margin: 0 0 12px; padding-left: 1.3em; }
li { margin-bottom: 6px; }
a { color: var(--dev-ink); }
code { font: 0.9em var(--mono); }
mark { padding: 0 3px; border-radius: 3px; font-style: normal; color: var(--john-ink); background: var(--john-bg); font-weight: 600;
  box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.tablewrap { overflow-x: auto; max-width: 100%; margin: 4px 0 14px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule); background: var(--surface); }
th, td { text-align: left; vertical-align: top; padding: 6px 10px 6px 6px; border-bottom: 1px solid var(--rule); }
thead th { font: 600 12px var(--font); color: var(--ink2); vertical-align: bottom; border-bottom: 1px solid var(--ink); }
td { font-variant-numeric: tabular-nums; }
.fig { margin: 18px 0 22px; max-width: 520px; display: grid; gap: 10px; }
.figchart { min-width: 0; }
.take { font: 600 19px/1.25 var(--head); text-wrap: balance; margin-bottom: 6px; }
.cap { font-size: 12.5px; line-height: 1.4; color: var(--muted); margin: 6px 0 0; }
svg.chart { width:100%; height:auto; display:block; background: var(--surface); border:1px solid var(--rule); border-radius:3px; }
svg .grp { font: 600 7.6px var(--font); fill: var(--ink2); stroke: var(--surface); stroke-width: 3px; paint-order: stroke; stroke-linejoin: round; }
svg .rowl { font: 7.6px var(--font); fill: var(--ink); } svg .leakyl { font-style: italic; } svg .bold { font-weight: 600; }
svg .ptitle { font: 600 7.8px var(--font); fill: var(--ink); }
svg .axs { font: 7.5px var(--font); fill: var(--muted); }
svg .frame { fill: none; stroke: var(--ink2); stroke-width: .7; }
svg .diag { stroke: var(--muted); stroke-width: .8; stroke-dasharray: 1.5 2; }
svg .mk.test-h { fill: var(--surface); stroke: var(--test); stroke-width: 1.3; }
svg .grey-f { fill: var(--muted); }
.band { margin: 8px 0; display: grid; gap: 14px; grid-template-columns: 1fr; }
.cell { margin: 0; max-width: none; display: block; }
.cell .take { margin-bottom: 4px; }
.cell .cap { margin-top: 4px; }
.predwrap { margin: 4px 0 12px; }
table.pred { font-size: 12.5px; }
table.pred th, table.pred td { padding: 3px 6px 3px 2px; vertical-align: top; }
table.pred th { font: 600 11px var(--font); }
table.pred { table-layout: fixed; } table.pred col.c0 { width: 12pt; } table.pred col.c1 { width: 40%; } table.pred col.c2 { width: 22%; } table.pred col.c3 { width: 26%; } table.pred col.c4 { width: 8%; }
td.pid { color: var(--muted); font: 8px var(--mono); padding-top: 4px; } td.vc { text-align: center; padding-right: 2px; }
.vm { display: inline-block; width: 8px; height: 8px; border-radius: 50%; border: 1.3px solid var(--ink); vertical-align: -1px; box-sizing: border-box; }
.vm.in { background: var(--ink); } .vm.near { background: var(--surface); } .vm.out { background: var(--surface); position: relative; }
.vm.out::after { content: ""; position: absolute; left: -1px; right: -1px; top: 50%; border-top: 1.3px solid var(--ink); transform: rotate(-45deg); }
.vm.none { border: 0; width: 8px; height: 0; border-top: 1.5px solid var(--ink); border-radius: 0; vertical-align: 3px; }
.key, .predline { font-size: 12px; color: var(--ink2); margin: 4px 0 0; }
svg .tag { font: 8px var(--font); }
svg .tick { font: 7.5px var(--mono); fill: var(--muted); }
svg .zerot { fill: var(--ink); font-weight: 600; }
svg .axt { font: 7.5px var(--font); fill: var(--ink2); }
svg .val { font: 600 7.5px var(--mono); fill: var(--ink); }
svg .dev-t { fill: var(--dev-ink); } svg .test-t { fill: var(--test-ink); }
svg .grid { stroke: var(--rule); stroke-width: .6; }
svg .zero { stroke: var(--ink); stroke-width: .9; }
svg .ci { stroke-width: 1.5; stroke-linecap: butt; }
svg .dev-s { stroke: var(--dev); } svg .test-s { stroke: var(--test); }
svg .mk { stroke: var(--surface); stroke-width: .9; }
svg .dev-f { fill: var(--dev); } svg .test-f { fill: var(--test); }
@media (min-width: 700px) { .band { grid-template-columns: 268fr 244fr; } }
a:focus-visible { outline: 2px solid var(--dev); outline-offset: 2px; }
.probe { height: 1.45em; }
@media print {
  :root:root:root { @LIGHT@ color-scheme: light; }
  @page { size: A4; margin: 8mm 12mm 9mm; }
  html { font-size: 8.6pt; }
  body { background:#fff; font-size: 8.6pt; line-height: 1.42; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .page { max-width: none; padding: 0; }
  .hero { padding-bottom: 5pt; margin-bottom: 8pt; display:flex; justify-content: space-between; align-items: baseline; gap: 12pt; flex-wrap: wrap; }
  h1 { font-size: 19pt; }
  .byline { margin: 0; font-size: 8.4pt; }
  .cols { column-count: 2; column-gap: 7mm; column-rule: 0.5pt solid var(--rule); }
  .cols > * { orphans: 2; widows: 2; }
  p { margin: 0 0 5pt; hyphens: auto; -webkit-hyphens: auto; hyphenate-limit-chars: 7 3 3; }
  .lede { font-size: 9.2pt; line-height: 1.42; hyphens: none; -webkit-hyphens: none; padding-bottom: 6pt; border-bottom: 0.5pt solid var(--rule); margin-bottom: 6pt; }
  h2 { font-size: 11.5pt; margin: 10pt 0 3pt; break-after: avoid; break-inside: avoid; }
  .colbreak { break-before: column; height: 0; }
  h3 { font-size: 9.5pt; margin: 6pt 0 2pt; break-after: avoid; }
  ul, ol { margin: 0 0 4.5pt; padding-left: 11pt; }
  li { margin-bottom: 3pt; hyphens: auto; -webkit-hyphens: auto; hyphenate-limit-chars: 7 3 3; }
  .tablewrap { overflow: visible; margin: 2pt 0 6pt; break-inside: avoid; }
  table { font-size: 8.4pt; line-height: 1.3; }
  th, td { padding: 3pt 4pt 3pt 2pt; }
  thead th { font-size: 8.4pt; }
  tr { break-inside: avoid; }
  .fig { max-width: none; margin: 8pt 0 8pt; column-span: all; break-inside: avoid; grid-template-columns: 1.2fr 1fr; gap: 14pt; align-items: center;
    padding: 8pt 0; border-top: 0.5pt solid var(--rule); border-bottom: 0.5pt solid var(--rule); }
  .take { font-size: 12pt; margin-bottom: 5pt; }
  .cap { font-size: 8.4pt; line-height: 1.35; margin: 0; }
  svg.chart { border: 0.5pt solid #c9d0d7; }
  mark { hyphens: none; -webkit-hyphens: none; border: 0.5pt dashed var(--john-ink); padding: 0 1.5pt; }
  .band { column-span: all; grid-template-columns: 268fr 244fr; gap: 15pt; margin: 6pt 0 6pt; padding: 6pt 0 5pt; break-inside: avoid;
    border-top: 0.5pt solid var(--rule); border-bottom: 0.5pt solid var(--rule); align-items: start; }
  .cell .take { font-size: 10pt; line-height: 1.2; margin-bottom: 3pt; min-height: 24pt; }
  .cell .cap { font-size: 7.6pt; line-height: 1.3; margin-top: 3pt; }
  svg.chart { border: 0; background: transparent; }
  .predwrap { margin: 2pt 0 5pt; break-inside: avoid; column-span: all; }
  table.pred { font-size: 7.8pt; line-height: 1.22; }
  table.pred th, table.pred td { padding: 1.6pt 3pt 1.6pt 1pt; }
  table.pred th { font-size: 7.6pt; }
  td.pid { font-size: 7.5pt; padding-top: 1.8pt; }
  .vm { width: 6.5pt; height: 6.5pt; vertical-align: -0.8pt; }
  .vm.none { width: 6.5pt; }
  .key, .predline { font-size: 7.8pt; line-height: 1.3; margin: 2pt 0 0; }
}
""".replace("@LIGHT@", LIGHT).replace("@DARK@", DARK)

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">'
)


def render_table(head, rows, aligns):
    def cell(tag, c, k):
        a = f' style="text-align:{aligns[k]}"' if aligns and k < len(aligns) and aligns[k] != "left" else ""
        return f"<{tag}{a}>{inline(c)}</{tag}>"

    return ('<div class="tablewrap"><table><thead><tr>' + "".join(cell("th", c, k) for k, c in enumerate(head))
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(r)) + "</tr>" for r in rows)
            + "</tbody></table></div>")


def build_html(probe: int = 0) -> tuple[str, str]:
    blocks = parse_blocks(MD.read_text(encoding="utf-8"))
    data = figure_data()
    cache: dict = {}
    getd = lambda k, fn: cache.setdefault(k, fn())
    title, byline, head_html, body = "Two-page report", "", "", []
    i = 0
    if blocks and blocks[0][0] == "h1":
        title = blocks[0][1]; i = 1
        if i < len(blocks) and blocks[i][0] == "p":
            byline = blocks[i][1]; i += 1
    lede_done, used_fig = False, False
    rest = blocks[i:]
    k = 0
    while k < len(rest):
        kind, val = rest[k]
        k += 1
        if kind == "figure" and val == "forest" and k < len(rest) and rest[k] == ("figure", "curves"):
            k += 1
            body.append('<section class="band">' + forest_cell(getd("forest", forest_data)) + curves_cell(getd("curves", curves_data)) + "</section>")
            continue
        if kind == "figure" and val in ("forest", "curves"):
            cell = forest_cell(getd("forest", forest_data)) if val == "forest" else curves_cell(getd("curves", curves_data))
            body.append('<section class="band" style="grid-template-columns:1fr">' + cell + "</section>")
            continue
        if kind == "tablemark":
            if val == "predictions":
                body.append(predictions_html(getd("pred", predictions_data)))
            else:
                body.append(f'<p><mark class="john">[[table "{html.escape(val)}" is not defined in build_two_page.py]]</mark></p>')
            continue
        if kind == "p":
            if not lede_done and not body:
                body.append(f'<p class="lede">{inline(val)}</p>'); lede_done = True
            else:
                body.append(f"<p>{inline(val)}</p>")
        elif kind in ("h1", "h2", "h3", "h4"):
            tag = "h2" if kind in ("h1", "h2") else "h3"
            body.append(f"<{tag}>{inline(val)}</{tag}>")
        elif kind in ("ul", "ol"):
            body.append(f"<{kind}>" + "".join(f"<li>{inline(x)}</li>" for x in val) + f"</{kind}>")
        elif kind == "table":
            body.append(render_table(*val))
        elif kind == "break":
            body.append('<div class="colbreak"></div>')
        elif kind == "figure":
            if val == "precedent":
                body.append(figure_html(data)); used_fig = True
            else:
                body.append(f'<p><mark class="john">[[figure "{html.escape(val)}" is not defined in build_two_page.py]]</mark></p>')
    if not used_fig:
        print("WARNING: no {{figure: precedent}} marker in two_page.md; the figure is not placed.", file=sys.stderr)
    body.extend('<div class="probe"></div>' for _ in range(probe))
    head_html = f'<header class="hero"><h1>{inline(title)}</h1><div class="byline">{inline(byline)}</div></header>'
    plain_title = re.sub(r"<[^>]+>", "", inline(title))
    doc = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '<meta name="claude:viewport-fit-sides" content="cover">'
        f"<title>{plain_title}</title>{FONTS}<style>{CSS}</style></head><body><div class=\"page\">"
        f'{head_html}<main class="cols">{"".join(body)}</main></div></body></html>\n'
    )
    return doc, plain_title


# ====================================================================== pdf + page count
def make_pdf(html_path: Path, pdf_path: Path):
    with tempfile.TemporaryDirectory() as prof:
        cmd = [CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
               "--virtual-time-budget=15000", "--no-pdf-header-footer",
               f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not pdf_path.exists():
        raise SystemExit(f"Chrome did not write the PDF.\n{r.stderr[-600:]}")


def count_pages(pdf_path: Path) -> tuple[int, str]:
    try:
        from pypdf import PdfReader
        rd = PdfReader(str(pdf_path))
        fonts = set()
        for pg in rd.pages:
            for f in (pg.get("/Resources", {}).get("/Font", {}) or {}).values():
                fonts.add(str(f.get_object().get("/BaseFont", "")))
        return len(rd.pages), ",".join(sorted(fonts))
    except ImportError:
        data = pdf_path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page[^s]", data)), "(pypdf not installed; fonts not checked)"


def main():
    probe = 0
    if "--probe" in sys.argv:
        probe = int(sys.argv[sys.argv.index("--probe") + 1])
    out_dir = HERE if not (probe or "--md" in sys.argv) else Path(tempfile.mkdtemp(prefix="two_page_probe_"))
    doc, _ = build_html(probe)
    hp, pp = out_dir / "two_page.html", out_dir / "two_page.pdf"
    hp.write_text(doc, encoding="utf-8")
    make_pdf(hp, pp)
    n, fonts = count_pages(pp)
    print(f"wrote {hp} and {pp}")
    if "Plex" not in fonts and "pypdf" not in fonts:
        print("WARNING: IBM Plex is not embedded in the PDF (Google Fonts unreachable?). Page count may differ.", file=sys.stderr)
    print(f"PDF pages: {n}" + (f"  (probe: +{probe} lines)" if probe else ""))
    if n != 2:
        print(f"ERROR: the PDF is {n} pages, not 2. "
              + ("The text has grown past two pages; shorten it." if n > 2 else "The text now fills fewer than two pages."),
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
