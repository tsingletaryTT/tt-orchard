# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A v6 bundle's run.sh in the current form, for tests that build a stub bundle.

Stages 2 and 4 choose a bundle only when stage 2's swap can rewrite its run.sh (orchard/weights_source.py, issue 73), so a stub that is
only `#!/bin/bash` is no longer a bundle that can take swapped weights. This is the smallest script the swap's own edit accepts."""


def valid_run_sh(repo: str) -> str:
    return f'''#!/usr/bin/env bash
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYBIN="$HERE/venv/bin/python"
export HF_MODEL="${{HF_MODEL:-{repo}}}"
"$PYBIN" "$HERE/prepare_model_dir.py"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "$HERE/model-dir" --max_num_seqs 4 "$@")
exec "${{CMD[@]}}"
'''
