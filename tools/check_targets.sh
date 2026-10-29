#!/bin/zsh
# Compiles every compute entry of every shader for the three backends' targets
# (Metal, CUDA, SPIR-V) with slangc and lists the failures. What can be checked
# for the Linux backends from a Mac; the GPU tests on real CUDA/Vulkan devices
# are the rest. Usage: tools/check_targets.sh [metal cuda spirv]
set -u
cd "$(dirname "$0")/.."
SLANGC=${SLANG_ROOT:-$HOME/tools/slang}/bin/slangc
targets=(${@:-metal cuda spirv})
tmp=$(mktemp -d)
fail=0; total=0
for f in shaders/sjxl/**/*.slang; do
  entries=($(awk '/\[shader\("compute"\)\]/{getline; while ($0 !~ /void /) getline; sub(/.*void /,""); sub(/\(.*/,""); print}' $f))
  for e in $entries; do
    for t in $targets; do
      total=$((total+1))
      out=$($SLANGC $f -I shaders -target $t -entry $e -stage compute -o $tmp/out.$t 2>&1)
      if [ $? -ne 0 ]; then fail=$((fail+1)); echo "FAIL $t $f:$e"; echo "$out" | grep -E "error" | head -3; fi
    done
  done
done
rm -rf "$tmp"
echo "done: $total compiles, $fail failures"
[ $fail -eq 0 ]
