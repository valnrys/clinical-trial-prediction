"""Why did a trial stop? A keyword reading of the sponsor's free-text "why stopped".

A trial can fail in two ways. It can stop before it delivers a result (often because
it could not recruit), or it can deliver a result that is negative. Our label only
sees the second kind. This module sorts the first kind by the reason the sponsor
gave, so the report can say how much of it there is.

The rules are fixed keywords, tried in order; the first category that matches wins:
stopped early for benefit, recruitment, efficacy or futility, safety, funding or
business, then "other". An empty reason is "no reason given". Order matters:
"sponsor decision based on slow recruitment" is a recruitment stop, not a business one.

Four things keep the rules from firing on a word alone:

- Negated phrases are removed first: "not due to any safety concerns", "unrelated to
  safety or efficacy", "no safety issues". Business stops often say what they were not.
- Recruitment needs a failure word next to the recruitment word ("slow enrolment",
  "unable to recruit", "few patients enrolled"). "Enrollment stopped due to toxicity"
  is a safety stop, not a recruitment one.
- "Interim analysis" is not a reason. It counts as efficacy or futility only with a
  failure word beside it, and as stopped early for benefit only with a benefit word.
- Short stems are word-bound, so "fundamental" is not funding and "company" is not
  "accompany".

The rules are crude and they were checked by hand twice. The first version was read on
40 stopped trials and 5 were wrong. The rules were then rewritten; the second version
was read on 60 fresh stopped trials, and its error rate is reported in
results/dev_failures.md. That count is the error rate of these rules: they were not
adjusted after the read.
"""

from __future__ import annotations

import re

NO_REASON = "no reason given"
BENEFIT = "stopped early for benefit"
RECRUITMENT = "recruitment"
EFFICACY = "efficacy or futility"
SAFETY = "safety"
BUSINESS = "funding or business"
OTHER = "other"

REASONS = (BENEFIT, RECRUITMENT, EFFICACY, SAFETY, BUSINESS, OTHER, NO_REASON)


def _any(*parts: str) -> str:
    """The parts as one regex that matches any of them."""
    return "|".join(parts)


_SAFE = r"(?:safety|toxic\w*|adverse|harm\w*|side[ -]effects?)"
_SAFE_OR_EFFICACY = r"(?:safety|efficacy|futility|toxic\w*|adverse|harm\w*|side[ -]effects?)"
_CAUSE = (
    r"(?:due to|because of|based on|related to|relating to|linked to|driven by|prompted by"
    r"|motivated by|caused by|the result of|a result of|result of|resulting from|owing to"
    r"|secondary to|attributable to|concerning|regarding|involving|for|over|by|on)"
)
_ADVERB = (
    r"(?:(?:directly|solely|only|primarily|mainly|simply|entirely|wholly|necessarily|really"
    r"|at all|in any way)\s+)?"
)
_NOT_BENEFIT = r"(?!benefit|efficacy|effect)"
_MORE = rf"(?:\s*(?:/|,|and/or|and|or|&)\s*(?:any\s+)?{_SAFE_OR_EFFICACY})*"

# Negated phrases: ways a sponsor says what the stop was NOT about. Removed before
# matching. "No efficacy seen" is a real efficacy stop and is not removed.
_NEGATED = re.compile(
    _any(
        # "not due to any safety concerns", "not based on the safety results"
        rf"\b(?:not|never|nor)\s+{_ADVERB}{_CAUSE}\s+(?:\w+\s+){{0,2}}?{_SAFE_OR_EFFICACY}{_MORE}\b",
        # "unrelated to safety", "not related to efficacy", "non-safety related"
        rf"\b(?:un-?related to|not related to|not linked to|nothing to do with)\W+(?:\w+\W+){{0,3}}?{_SAFE_OR_EFFICACY}{_MORE}\b",
        r"\bnon-?\s?(?:safety|efficacy)\b",
        # "no safety issues", "no adverse events", "not safety, study drug, or adverse event"
        rf"\b(?:no|without|nor|neither|not)\s+(?:{_NOT_BENEFIT}\w+\s+){{0,2}}{_SAFE}{_MORE}\b",
        # "no safety or efficacy concerns", "no efficacy issues"
        rf"\bno\s+{_SAFE_OR_EFFICACY}(?:\s*(?:/|,|and|or|&)\s*{_SAFE_OR_EFFICACY})*\s+(?:concerns?|issues?|problems?|reasons?|grounds?)\b",
        # "safety was not a factor"
        rf"\b{_SAFE_OR_EFFICACY}\s+(?:\w+\s+){{0,2}}?(?:not|no)\s+(?:a\s+|the\s+|an\s+)?(?:factor|reason|issue|concern|cause|basis|consideration)\b",
        # "providing considerable safety data"
        r"\b(?:provid\w+|collect\w*|gather\w*|generat\w*|yield\w*)\s+(?:\w+\s+){0,2}safety\s+(?:data|information)\b",
    )
)

_ENROL = r"(?:enrol\w*|recruit\w*|accru\w*|inclusions?\b(?! criteria))"
_PEOPLE = r"(?:patients?|participants?|subjects?|individuals?|volunteers?|candidates?)"
_WEAK = (
    r"(?:slow\w*|poor\w*|low|insufficient|inadequate|limited|sluggish|disappointing|difficult\w*"
    r"|problems?|issues?|challeng\w*|lacking|unsatisfactory|below|less than|fewer than|lower than"
    r"|behind|standstill|stalled|too (?:few|slow|low))"
)
_GAP = r"(?:(?!due\b|because\b|stopped\b|halted\b|terminated\b|closed\b|paused\b|suspended\b|hold\b)\w+\s+)"

_RECRUITMENT = re.compile(
    _any(
        # "slow enrolment", "poor accrual", "low patient recruitment"
        rf"\b{_WEAK}\s+(?:\w+\s+){{0,3}}?{_ENROL}\b",
        # "enrolment was slow", "recruitment lower than expected", "accrual problems"
        rf"\b{_ENROL}\s+{_GAP}{{0,4}}?{_WEAK}\b",
        # "lack of patients", "lack of enrolment", "not enough participants", "too few subjects"
        rf"\black of\s+(?:eligible\s+)?(?:{_PEOPLE}|{_ENROL}|interest from {_PEOPLE})",
        rf"\b(?:not enough|too few|insufficient number of|inadequate number of)\s+(?:eligible\s+)?(?:{_PEOPLE}|{_ENROL})",
        # "unable to recruit", "failure to enrol", "failed to meet enrollment goals"
        rf"\b(?:unable to|inability to|failure to|failed to|not able to|difficult\w*(?: in| to)?|difficulty(?: in)?)\s+(?:\w+\s+){{0,2}}?{_ENROL}\b",
        rf"\b(?:unable to|inability to|could not|couldn't|cannot|difficult\w* (?:in )?)\s*(?:find|identify|obtain|meet|reach)\b[^.;]{{0,30}}(?:{_PEOPLE}|{_ENROL}|target|goal|sample size)",
        # "did not enroll enough", "did not meet target accrual", "not possible to recruit"
        rf"\b(?:did not|didn't)\s+(?:\w+\s+)?{_ENROL}\s+(?:enough|sufficient|adequate)",
        rf"\b(?:not|never|unable to|failed to|cannot|could not|couldn't)\s+(?:\w+\s+){{0,3}}?(?:reach|meet|achieve|attain|hit)\w*\s+(?:\w+\s+){{0,3}}?{_ENROL}",
        rf"\b(?:not possible to|impossible to|not feasible to|could not be|couldn't be|could not|couldn't)\s+(?:\w+\s+)?{_ENROL}",
        rf"\b{_ENROL}\s+(?:was |is |were )?(?:not|never)\s+(?:possible|feasible|achieved|achievable)",
        r"\bnot\s+(?:\w+\s+)?(?:interested|willing)\s+(?:in|to)\s+(?:enrol|particip|join|take part)",
        # a short reason that opens with "no" or "none" and is about enrolment: "No enrollment."
        rf"^\W*(?:no|none|zero|0)\b[^.;]{{0,40}}?\b{_ENROL}[^.;]{{0,40}}\W*$",
        # "few patients enrolled", "only 3 patients recruited"
        rf"\b(?:few|fewer)\s+(?:eligible\s+)?{_PEOPLE}\s+(?:\w+\s+){{0,2}}(?:enrolled|recruited|accrued|eligible|identified|available)",
        rf"\bonly\s+\d+\s+{_PEOPLE}\s+(?:\w+\s+){{0,4}}(?:enrolled|recruited|accrued)",
        # "no eligible patients", "no patients could be recruited"
        rf"\bno\s+eligible\s+{_PEOPLE}",
        rf"\bno\s+{_PEOPLE}\s+(?:could|were able to)\s+be\s+(?:\w+\s+)?(?:enrolled|recruited|identified|found)",
        # "enrollment target not met", "accrual goals were not met"
        rf"\b{_ENROL}\s+(?:\w+\s+){{0,2}}(?:targets?|goals?|numbers?|rates?)\s+(?:\w+\s+){{0,2}}?(?:not|never)\s+(?:\w+\s+)?(?:met|reached|achieved|attained)",
        # "due to recruitment" when the recruitment word is itself the reason
        rf"\b(?:due to|because of|owing to|secondary to)\s+(?:\w+\s+)?{_ENROL}\b(?!\s+(?:of|was|were|had|has|hold|pause|halt|suspension|freeze)\b)",
        # the whole reason is a recruitment noun phrase: "Accrual", "Recruitment issues"
        rf"^\W*(?:patient |subject |participant )?{_ENROL}(?: (?:issues?|problems?|difficulties|challenges?|rate|failure|shortfall))?\W*$",
    )
)

_FAILURE_RE = re.compile(
    r"futil\w*|no (?:\w+ ){0,2}(?:significant|benefit|difference|effect|efficacy)"
    r"|not (?:\w+ ){0,2}(?:meet|met|show|shown|demonstrat\w*|support\w*|likely|reach\w*|achiev\w*|confirm\w*|reveal\w*)"
    r"|did ?n[o']t|unlikely|fail\w*|insufficient|lack\w*|negative|boundar\w*|stopping rule"
    r"|low (?:probab\w*|likel\w*|chance|power)|crossed|inferior|worse|detriment\w*"
)
_INTERIM = re.compile(
    r"\b(?:interim|intermediary)\s+(?:analys[ei]s|results?|review|look|data|evaluation|assessment)"
)
_EFFICACY = re.compile(
    _any(
        r"futility|efficacy|lack of (?:effect|response|benefit)|ineffective|not effective",
        r"did not meet (?:the )?(?:primary|co-primary|study endpoint|endpoint|objective|criteria|expectation)",
        r"results?,? (?:were |was )?negative|negative results?",
        r"response rate|no (?:additional |clinical )?benefit",
        r"low (?:conditional )?(?:probability|likelihood|chance|power) (?:of|to) (?:\w+ ){0,3}(?:success|benefit|significan\w*|positive|demonstrat\w*|efficacy)",
        # "no significant difference between arms", "did not show significant benefit"
        r"\bno (?:\w+ )?significant (?:difference|benefit|effect|improvement|advantage)",
        (
            r"\b(?:did not|didn't|failed to|not|unlikely to)\s+(?:\w+\s+){0,2}"
            r"(?:show|demonstrate|reveal|achieve|attain|detect)\w*\b[^.;]{0,30}"
            r"(?:significant|benefit|difference|effect)"
        ),
        # three the first hand read missed on the wider data
        r"non-?inferiority[^.;]{0,50}(?:not (?:been )?(?:achieved|met|shown|demonstrated)|fail\w*|unlikely|could not)",
        r"not (?:sufficient|adequate|enough)\b[^.;]{0,25}(?:efficacy|effect|benefit|response|activity)",
        r"benefit[/ -]*(?:to[ -])?risk[^.;]{0,30}(?:does not|do not|doesn't|not|no longer|unfavou?rable|negative)",
    )
)

_SAFETY = re.compile(r"safety|toxic|adverse|tolerab|\bharm(?:s|ful)?\b")
_BUSINESS = re.compile(
    r"\bfund(?:s|ing|ed|ers?)?\b|sponsor(?:'s)? decision|business|strateg|\bcompan(?:y|ies)\b|financ"
    r"|commercial|priorit|portfolio|acquisition"
)

_BENEFIT = re.compile(
    _any(
        r"overwhelming (?:efficacy|benefit)",
        r"superiority",
        r"positive (?:results?|outcomes?|interim|findings?|data)",
        r"statistically significant (?:benefit|treatment effect|improvement|difference|advantage)",
        r"(?:met|achieved|reached) (?:the |its )?(?:primary|efficacy)",
        r"efficacy (?:was |had been )?(?:clearly |already )?(?:demonstrated|established|proven|shown)",
        r"demonstrated (?:clear |significant |sufficient )?(?:efficacy|benefit)",
        r"early (?:for|due to|because of) (?:efficacy|benefit)",
        r"clear benefit",
    )
)
_DENIAL_BEFORE = re.compile(
    r"\b(?:no|not|non|without|nor|neither|lack of|fail\w*|did ?n[o']t|unable|cannot|can't"
    r"|unlikely|low|insufficient|too|would not|will not|if)\b"
)
_DENIAL_AFTER = re.compile(r"\b(?:not|n't|unlikely|fail\w*|cannot|couldn't|unconfirmed)\b")


def _stopped_for_benefit(text: str) -> bool:
    """A benefit phrase that is not itself denied ("no superiority", "unlikely to show benefit")."""
    for match in _BENEFIT.finditer(text):
        before = re.split(r"[.;]", text[max(0, match.start() - 60) : match.start()])[-1]
        after = re.split(r"[.;]", text[match.end() : match.end() + 25])[0]
        if not _DENIAL_BEFORE.search(before) and not _DENIAL_AFTER.search(after):
            return True
    return False


def _futility_or_efficacy(text: str) -> bool:
    """An efficacy or futility phrase, or an interim analysis described with a failure word."""
    return bool(_EFFICACY.search(text) or (_INTERIM.search(text) and _FAILURE_RE.search(text)))


def classify_stop_reason(why_stopped: str | None) -> str:
    """One of REASONS for a free-text "why stopped" (None or blank is "no reason given")."""
    if why_stopped is None or not why_stopped.strip():
        return NO_REASON
    text = _NEGATED.sub(" ", why_stopped.lower())
    if _stopped_for_benefit(text):
        return BENEFIT
    if _RECRUITMENT.search(text):
        return RECRUITMENT
    if _futility_or_efficacy(text):
        return EFFICACY
    if _SAFETY.search(text):
        return SAFETY
    if _BUSINESS.search(text):
        return BUSINESS
    return OTHER
