#!/bin/zsh
# Fetch the Clips Core ML package, its tokenizer and metadata, and a whisper.cpp model into models/.
# Clips weights: Desert Ant Labs Source-Available License 1.0 (https://license.desertant.com/1.0).
set -eu
root=${0:A:h:h}/models
clips=https://huggingface.co/desert-ant-labs/clips/resolve/main
whisper=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
whisper_model=${WHISPER_MODEL:-ggml-base.en.bin}

mkdir -p "$root"
staging=$(mktemp -d "$root/.fetch.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Explicit exit ensures zsh runs the EXIT trap after a function fails.
fetch() { curl -L --fail --silent --show-error --create-dirs -o "$staging/$1" "$2/$1" || exit $?; }

clips_files=(
  clips.mlmodelc/analytics/coremldata.bin
  clips.mlmodelc/coremldata.bin
  clips.mlmodelc/metadata.json
  clips.mlmodelc/model.mil
  clips.mlmodelc/weights/weight.bin
  clip_tokenizer.bin
  clips_meta.json
)
for f in "${clips_files[@]}"
do fetch "$f" "$clips"; done
fetch "$whisper_model" "$whisper"

echo "f14166c0b746be5e4581954bfbbf0ec04b54c1eae3485f7abd8f62aa28a38fb4  $staging/clip_tokenizer.bin" | shasum -a 256 -c -
# Publish only after every download and the checksum succeed.
for f in "${clips_files[@]}" "$whisper_model"; do
  mkdir -p "$root/${f:h}"
  mv -f -- "$staging/$f" "$root/$f"
done
du -sh "$root"/*
