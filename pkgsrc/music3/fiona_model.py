# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runner file of the minimax-music3-fiona-crapple thin bundle.

`tt-model package-thin` wants a runner file at the bundle root (`--model-py`). For kind `tt-dit-server` nothing imports it: `run.sh` serves
`fiona_asgi:app` with uvicorn. The ASGI module lives in `extra_code/` and not here, because install.sh's `--verify` statements run with only
`extra_code/` on the path. This file re-exports the app so that `fiona_model:app` names the same thing.
"""
from fiona_asgi import app  # noqa: F401
