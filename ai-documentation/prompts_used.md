# Key Prompts Used — Emma Rogoveanu (Claude Code)

Prompt-engineering examples, as the course AI-use instructions require. Each prompt
below is an **excerpt copied verbatim** from Claude Code's session logs. The full
prompts and replies are in the transcript named in each heading. Times are
America/Phoenix. Komal's prompts are in
[komal-streamlit-session-01.md](komal-streamlit-session-01.md).

## 1. Setting the working rules (architecture, data, code generation)

Oct 3, 15:26, [emma-claude-session-2026-10-03.md](emma-claude-session-2026-10-03.md)

> You are helping me build my CST-435 Topic 2 group project, “Income Insight.” …
> DO NOT modify the professor’s reference template. Only modify my working repository.

**Technique:** fixed the scope and the off-limits files before any code was written.

## 2. Keeping control of execution

Oct 3, 16:36, [emma-claude-session-2026-10-03-pt2.md](emma-claude-session-2026-10-03-pt2.md)

> You are my coder. I will personally run commands, test code, use Supabase, commit/push,
> and deploy so I understand what I am doing.

**Technique:** separated code generation (AI) from execution and verification (Emma).
Later step prompts repeated this rule.

## 3. Making the design decisions myself (preprocessing)

Oct 3, 16:38, [emma-claude-session-2026-10-03-pt2.md](emma-claude-session-2026-10-03-pt2.md)

> 1. **Drop `education` from the model features and keep `education_num`.** …
> 3. **Do not use `sex` or `race` as model inputs.** Keep both in `adult_income` for
>    fairness auditing … while still acknowledging in the model card that other features
>    such as `relationship`, `occupation`, and `marital_status` can act as proxies …
> 4. **Do not add `log1p` yet.** …
> 5. **Use `"Unknown"` for missing categorical values** rather than the most common category.

**Technique:** approved the AI's proposal only with explicit decisions and reasons.

## 4. Tightening an experimental control

Oct 3, 17:03, [emma-claude-session-2026-10-03-pt3.md](emma-claude-session-2026-10-03-pt3.md)

> Change the third config so it isolates architecture rather than architecture + dropout:
> - name: `deep` … dropout: 0.1
> Keep all other controlled values identical: …

**Technique:** corrected a confounded comparison, where the original third config
changed two variables, so that each config differs from `baseline` in one thing only.

## 5. A one-off, guarded data script (debugging and safety)

Oct 3, 17:51, [emma-claude-session-2026-10-03-pt4.md](emma-claude-session-2026-10-03-pt4.md)

> write a one-off script that reads models/gelu/test_predictions.csv, verifies all 7,327
> rows are test rows, attaches served_by_run_id = 3, inserts them into predictions,
> refuses duplicates, and then verifies v_fairness_audit returns sex-group FPR/FNR rows.

**Technique:** stated the invariants the script must check before it writes.

## 6. Rubric-driven fixes with an honesty constraint

Oct 4, 14:19, [emma-claude-session-2026-10-04-current.md](emma-claude-session-2026-10-04-current.md) (prompt 1)

> Inspect and fix the remaining CST-435 Topic 2 Income Insight app gaps in this
> repository. Implement the changes, verify them, and leave them ready for review. Do not
> commit, push, or deploy.

## 7. Final audit (this audit)

Oct 4, 16:58, [emma-claude-session-2026-10-04-current.md](emma-claude-session-2026-10-04-current.md) (prompts 9–10)

> Perform a final evidence-based audit and fix … Distinguish verified, user-confirmed,
> unverified, and missing evidence. Do not promise a grade.

**Technique:** asked for evidence categories, and for verification to be kept separate
from claims, instead of asking for "everything meets Target".

## Not included here

The prompts Emma drafted in ChatGPT/Codex are not captured yet. See
[emma-chatgpt-codex-assistance.md](emma-chatgpt-codex-assistance.md).
