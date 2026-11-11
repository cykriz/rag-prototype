---
description: Blind code review of all uncommitted changes via the review-changes subagent
---

Run one blind code review of the working tree.

1. **Delegate, do not review.** Call the Agent tool with `subagent_type: "review-changes"` and exactly this
   prompt, nothing else — no task context, no intent, no hints:

   Review all uncommitted changes

   Wait for its result before step 2. Do not review the diff yourself, before or after.

2. **Verify every finding, apply only within bounds.** Read the cited location; for duplication read both
   sides. Drop what does not hold. Apply a finding automatically only if it is `proven`, `high` or `medium`,
   and its fix touches only files in the **allowed set**:
   - started by the Stop hook → the "Files changed this turn" list from its reason;
   - started manually (`/review`, e.g. before a commit) → all uncommitted files.

   Everything else — `conjecture`, `low`, or a fix outside the allowed set — is not applied and stays open.

3. **Run the gates.** Always run every command under `GATES` yourself (lint, typecheck, test), whether or not
   fixes were applied.

4. **Report briefly.** One status line —
   `Review: <verdict> · <n> applied · <n> dropped · gates <pass|fail>` — then only the findings still open
   (not applied, not dropped). Name dropped ones by location on one line, no reasons. In a Stop-hook run, open
   findings outside the allowed set are only counted in the status line (`<n> outside this turn`).

Never commit, push or stage.
