# Where the files in pkgsrc/music3 come from

Nothing here was written for this repository except `fiona_asgi.py`, `fiona_model.py`, `fiona-README.md` and `fiona-NOTICE`.

| path | source | revision |
| --- | --- | --- |
| `code/models/` | `jashansinghTT/minimax-music3-blackhole`, directory `code/`. Only the 63 files the server needs: the whole `models/autoports/minimaxai_minimax_music3` directory, every `models/...` module that the server imports (found by following the import statements from `server/app.py`), `models/tt_transformers/tt/prefetcher/prefetcher_config.yaml` and `models/common/LICENSE-Llama-3.1`. The files are unchanged. | `5a789cd8099cec3a6dc4386db3b4a1ae3925504c` |
| `demo/music_page.py` | `episod/minimax-music3-fiona-crapple-blackhole`, `demo/app.py`, renamed because `app` is the name of the ASGI attribute. Unchanged. | `2a2ae57a63961fc13cbe1fcfac4f4f5360f1cae3` |
| `demo/music_client.py` | the same repository, `demo/music_client.py`. Unchanged. | `2a2ae57a63961fc13cbe1fcfac4f4f5360f1cae3` |

The model code is Apache-2.0 (SPDX headers in the files; the reference ports come from huggingface/diffusers, see
`code/models/autoports/minimaxai_minimax_music3/reference/PROVENANCE.md`). The Hub package says its `code/` is byte-identical to the model code inside its
container image, and gives the digest `b131ad02856e6ea7...` of the whole directory; the directory here is a subset, so the digest does not apply.

To refresh the copy: download that repository at the new revision (read only), run the same selection, and compare the files with `diff -r`.
