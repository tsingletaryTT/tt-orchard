# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runner file of the Aion-1.0-Instruct-2.3B thin bundle.

`tt-model package-thin` wants a runner file at the bundle root (`--model-py`). For kind `tt-dit-server` nothing imports it: `run.sh` serves
`aion_asgi:app` with uvicorn. The ASGI module lives in `extra_code/` and not here, because install.sh's `--verify` statements run with only
`extra_code/` on the path. This file re-exports the app so that `aion_model:app` names the same thing.
"""
from aion_asgi import app  # noqa: F401
