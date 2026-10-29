#!/bin/bash
# Speed-only rounds, interleaved so that load from other jobs on the host lands on every mode alike.
# 3 traced rounds of the 7 configurations. Run as: MATRIX=matrix3.sh bash runlease2.sh OUTDIR
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
cfgs=("composed|--ssd composed" "sf-repeat|--ssd state_full --expand repeat" "sf-mul|--ssd state_full --expand mul"
      "sfro-repeat|--ssd state_full_ro --expand repeat" "sfro-mul|--ssd state_full_ro --expand mul" "bcast|--ssd bcast" "bcastro|--ssd bcast_ro")
args=()
for r in 1 2 3; do
  for c in "${cfgs[@]}"; do n=${c%%|*}; a=${c#*|}; args+=("$n-tr-r$r|$a --trace --no-agree"); done
done
for c in "${cfgs[@]}"; do n=${c%%|*}; a=${c#*|}; args+=("$n-un-r1|$a --no-agree"); done
bash $HERE/session2.sh $OUT "${args[@]}"
