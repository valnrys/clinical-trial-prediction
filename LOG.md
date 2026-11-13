# Log

The report must state active time spent, and where coding assistants helped
and where they cost time. Both are recorded here as they happen.

## Active time

Excludes waiting on downloads, data processing and agent runs.

**Total: 8 hours.** I did not keep a per-session record.

## Coding-assistant notes

Assistant used: Claude Code.

| Date | Helped / cost time | What happened |
|---|---|---|
| 10 Nov | Helped | Set up the repository, the held-out guard and the raw downloader before any design work. |
| 11 Nov | Helped | Opus and Gemini reviewed the report after the final run; it was revised after both (825ba44, f85a814). |
| 11 Nov | Cost time | The final plan had a sentence that tripped its own unfinished-section check; reworded before the run (69dd31b). |
| 11 Nov | Cost time | The report revision after review dropped two limits (forecast date, late results). Restored in f85a814. |
| 12 Nov | Helped | Four rounds of review before the second test run found real faults, among them model fits reused after one of their inputs had changed, and a dress rehearsal that did not follow the same path as the real run. All fixed before the run (441e314). |
| 12 Nov | Cost time | Agents left polling in loops had to be stopped by hand. |

## Final summary

**Active time:** 8 hours in total. This is the author's own time and excludes
waiting on agents and data processing.

**Coding assistants:** Claude Code. Sonnet wrote the code. Opus and Gemini
reviewed changes before they were committed.

| Helped / cost time | What happened |
|---|---|
| Helped | Breadth: a lot of work got done that would not have fitted in the time otherwise. |
| Cost time | The agents drifted into measurement for its own sake, rather than what the results mean. |
