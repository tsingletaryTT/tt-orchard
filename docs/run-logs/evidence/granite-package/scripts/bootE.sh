#!/bin/bash
# Boot the final staged package E from its own directory in the fresh venv, weights from the scratch cache. Short.
OUT=$1; R=$HOME/orchard-runs/granite-package
export HF_HOME=$R/fresh/hf-home TT_METAL_HOME=$HOME/tt-metal PORT=20110 GRANITE_VENV=$R/fresh/venv READY_TRIES=120 READY_SLEEP=2
PKG=$R/fresh/pkgE; rm -rf $PKG; cp -r $R/stage/E/granite-4.0-h-1b-blackhole $PKG
mkdir -p $OUT; ( source $GRANITE_VENV/bin/activate; cd $PKG; bash run.sh --self-test ) > $OUT/selftest.txt 2>&1; echo "selftest rc=$?"; tail -1 $OUT/selftest.txt
bash $R/boot.sh $PKG $OUT/boot 2>&1 | tail -14
( cd $PKG && sha256sum -c SHA256SUMS 2>&1 | grep -vc ": OK" | sed 's/^/files not OK: /' )
