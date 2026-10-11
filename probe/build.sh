#!/bin/zsh
# Build the Core ML probe with TinyCC. No Objective-C compiler and no -framework flags:
# the probe dlopens libobjc, Foundation and CoreML at run time.
set -eu
here=${0:A:h}
mkdir -p "$here/build"
tcc -Wall -o "$here/build/coreml_probe" "$here/coreml_probe.c"
