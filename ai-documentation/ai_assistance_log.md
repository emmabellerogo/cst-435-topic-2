# AI Assistance Log — Emma Rogoveanu

Integration notes required by the course AI-use instructions: what AI helped with,
how the output was adapted, and where Emma's decisions differed from what the AI
proposed. The rows are drawn from the Git history and the session transcripts
named in each row. **This file is a summary, not a transcript.** Komal's AI use is
recorded in [komal-streamlit-session-01.md](komal-streamlit-session-01.md).

| Area | Commit(s) | AI tool | What the AI produced | Emma's direction, changes and checks | Source |
|------|-----------|---------|----------------------|--------------------------------------|--------|
| Schema and loader | `49cba46` | Claude Code | `001_init.sql`, `db/load.py` | Required official UCI files, raw files saved under `data/raw/`, and a dry run before any insert. Ran the load herself | Oct 3 transcript |
| Preprocessing | `5fc0197` | Claude Code | `shared/features.py`, `api/preprocessing.py`, tests | Set the feature decisions herself: drop `education` and `fnlwgt`, exclude `sex`/`race`, no `log1p`, `"Unknown"` for missing values. See [prompts_used.md §3](prompts_used.md) | pt2 |
| Training pipeline | `cf7fb96` | Claude Code | config-driven trainer, early stopping | Ran training herself | pt2 |
| Controlled experiment | `72efe69` | Claude Code | `run_experiments.py`, three configs | **Deviated from the AI's draft:** replaced `deep_dropout` with `deep` so the third config changes only the architecture. See [§4](prompts_used.md) | pt3 |
| Calibration, test evaluation, persistence | `cc42073` | Claude Code | temperature scaling, one-time test evaluation, `persist_runs.py` | Ran evaluation and persistence herself, with dry runs first | pt4 |
| Serving API | `378cb7f` | Claude Code | frozen-model FastAPI service | Deployed it to Render herself | pt4 |
| Audit loader | `1c974b1` | Claude Code | `db/log_test_predictions.py` | Specified the invariants: test rows only, no duplicates, check the view | pt4 |
| UI fixes, README, model card, CI | `b2ec7b4`, `db4bbf6` | Claude Code | the changes listed in the Oct 4 summary | Reviewed and merged them, and checked the public deployment (Ⓔ in the README) | Oct 4 transcript |
| Final audit | uncommitted at time of writing | Claude Code | the numerical XOR backprop step, SQL query file, live `/predict` write test, CI trigger, report edits, `reports/final-rubric-audit.md` | Pending Emma's review | Oct 4 transcript, prompt 10 |
| Report draft, video script, prompt drafting | — | ChatGPT / Codex | drafts | **Not yet documented**; see [placeholder](emma-chatgpt-codex-assistance.md) | — |

## Where AI guidance was followed and where Emma deviated

- **Followed:** most generated code was kept as written after Emma reviewed it and ran
  the tests. The transcripts show each proposal and Emma's response. This log does
  not claim which individual design ideas originated with the AI.
- **Deviated or overrode:**
  - the feature-selection choices above
  - the controlled third configuration
  - the rule that the AI never runs commands, trains, writes to Supabase, commits or
    deploys in the Oct 3 sessions, which Emma set in every step prompt

## What I learned (Emma to write — not AI-generated)

_Emma: replace this line with your own reflection. The course asks what you learned
from using AI on this project. It should not be drafted by an AI tool._
