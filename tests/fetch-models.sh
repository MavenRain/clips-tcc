#!/bin/zsh
# Exercise download failures without network access or model weights.
set -euo pipefail
repo=${0:A:h:h}
test_root=$(mktemp -d "${TMPDIR:-/tmp}/clips-fetch-test.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT
mkdir -p "$test_root/bin"

cat > "$test_root/bin/curl" <<'STUB'
#!/bin/zsh
set -eu
output=
while (( $# )); do
  if [[ $1 == -o ]]; then output=$2; shift 2; else url=$1; shift; fi
done
count=0
[[ ! -f $CLIPS_FETCH_TEST_STATE ]] || count=$(<"$CLIPS_FETCH_TEST_STATE")
(( count += 1 ))
print -r -- "$count" > "$CLIPS_FETCH_TEST_STATE"
mkdir -p "${output:h}"
print -r -- "fresh ${url:t}" > "$output"
(( count != CLIPS_FETCH_TEST_FAIL_AT )) || exit 28
STUB
cat > "$test_root/bin/shasum" <<'STUB'
#!/bin/zsh
cat > /dev/null
exit "$CLIPS_FETCH_TEST_CHECKSUM_EXIT"
STUB
chmod +x "$test_root/bin/curl" "$test_root/bin/shasum"

files=(
  clips.mlmodelc/analytics/coremldata.bin
  clips.mlmodelc/coremldata.bin
  clips.mlmodelc/metadata.json
  clips.mlmodelc/model.mil
  clips.mlmodelc/weights/weight.bin
  clip_tokenizer.bin
  clips_meta.json
  ggml-base.en.bin
)

prepare() {
  case_root=$test_root/$1
  mkdir -p "$case_root/scripts" "$case_root/models"
  cp "$repo/scripts/fetch-models.sh" "$case_root/scripts/fetch-models.sh"
  for file in "${files[@]}" custom.bin; do
    mkdir -p "$case_root/models/${file:h}"
    print -r -- "cached $file" > "$case_root/models/$file"
  done
}

fetch_case() {
  env PATH="$test_root/bin:$PATH" \
    CLIPS_FETCH_TEST_STATE="$case_root/count" \
    CLIPS_FETCH_TEST_FAIL_AT="$1" CLIPS_FETCH_TEST_CHECKSUM_EXIT="$2" \
    zsh "$case_root/scripts/fetch-models.sh" > "$case_root/fetch.log" 2>&1
}

for fail_at in {1..8} 0; do
  prepare "failure-$fail_at"
  checksum_exit=0
  (( fail_at != 0 )) || checksum_exit=1
  if fetch_case "$fail_at" "$checksum_exit"; then
    print -u2 -- "FAIL: failure-$fail_at unexpectedly succeeded"
    exit 1
  fi
  for file in "${files[@]}" custom.bin; do
    if [[ $(<"$case_root/models/$file") != "cached $file" ]]; then
      print -u2 -- "FAIL: failure-$fail_at replaced cached $file"
      exit 1
    fi
  done
  remaining=( "$case_root/models"/.fetch.*(N) )
  (( ${#remaining} == 0 )) || {
    print -u2 -- "FAIL: failure-$fail_at left temporary downloads"
    cat "$case_root/fetch.log" >&2
    exit 1
  }
done

prepare success
for pass in 1 2; do
  rm -f "$case_root/count"
  fetch_case 0 0
  for file in "${files[@]}"; do
    [[ $(<"$case_root/models/$file") == "fresh ${file:t}" ]] || {
      print -u2 -- "FAIL: successful fetch $pass did not install $file"
      exit 1
    }
  done
  [[ $(<"$case_root/models/custom.bin") == 'cached custom.bin' ]]
done
prepare first-install
rm -rf "$case_root/models"
fetch_case 0 0
for file in "${files[@]}"; do
  [[ $(<"$case_root/models/$file") == "fresh ${file:t}" ]]
done
print -- 'PASS: eight partial downloads and a checksum failure preserve cached models; fresh and repeated fetches install all files'
