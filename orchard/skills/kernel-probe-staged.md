## Staged construction (this attempt builds the kernel in steps)

You do not write the whole kernel at once. The kernel is built in stages. Each stage has its own check, and you are given one stage at a time.

- The user message names the stage and says what `kernel.py` must do at the end of it.
- `kernel_spec.json` and `kernel_check.py` describe THIS stage only: its inputs, its output shape and what it computes. They change at every stage.
- From the second stage on, `kernel.py` already holds the kernel that passed the previous stage. Read it first. Change it as little as the new stage needs, and keep what already works.
- Do not write steps of later stages early. A stage is done when `kernel_check.py` prints `PASS`.
- If the stage's output shape differs from the previous stage's, the output buffer, the copy to `out` and the compute step have to agree on it.
- You get a limited number of turns and a limited time for each stage. When a stage's budget is spent the attempt ends at that stage.
