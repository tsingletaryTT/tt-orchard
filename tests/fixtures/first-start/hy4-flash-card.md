<!-- Excerpt of the package card of raahemnabeel/hy4-flash at revision 7e666138 (Prerequisites and Quickstart sections). Text as published; kept as a test fixture for the first-start check (issue 843). -->

## Prerequisites

- The [Tenstorrent CLI](https://github.com/tenstorrent/tt-cli/blob/main/docs/prerequisites.md), `tt`: `uv tool install tenstorrent` (or use tt-model alone, see the Quickstart)
- Docker
- Tenstorrent hardware: **p300x2**
- About 116 GB of free host RAM for full decode speed (17 GB minimum). With less than 116 GB free the server
  still starts but reads the Engram tables from disk and decodes more slowly. Below 17 GB it refuses to start
  (set `HY4_RAM_CHECK=warn` to start anyway).
- About 610 GB of disk: the 440 GB checkpoint plus about 165 GB of caches built on the first start.
- Optional: to keep the Engram tables from being swapped out under memory pressure, run the command that
  `tt-model serve --print` prints with `--ulimit memlock=-1:-1` added after `docker run`.

## Quickstart

```bash
tt serve raahemnabeel/hy4-flash
```

`tt serve` downloads the Docker image and the [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) at `b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f` weights (into your HF cache; they are not in the image), then starts an OpenAI-compatible server on port 20000 (or the next free port, if that one is busy). The first start compiles kernels for your device, which takes several minutes; the server is ready when it logs `Application startup complete`.

Without tt-cli: `tt-model serve raahemnabeel/hy4-flash`.

The first start also builds about 165 GB of caches from the checkpoint (the bfp4 experts and the FP8 Engram
tables) before the server is ready. Later starts are ready in about 4 minutes.

