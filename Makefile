# clips-tcc: plain C built with TinyCC (make 3.81 compatible).
# Each stage owns one fragment under mk/ and edits only that fragment:
#   OBJS += $(BUILD)/<module>.o          library objects linked into clips and the tests
#   LIBS += ...                          link flags for that module
#   CFLAGS_<module> = ...                extra compile flags for src/<module>.c
#   CHECKS += check-<stage>              gate targets run by `make check`
CC = tcc
CFLAGS = -Wall -Isrc
BUILD = build
OBJS =
LIBS =
CHECKS =

# Shared Unicode helpers (src/uni.c); every stage links them.
UTF8PROC = /opt/homebrew/opt/utf8proc
CFLAGS_uni = -I$(UTF8PROC)/include
OBJS += $(BUILD)/uni.o
LIBS += -L$(UTF8PROC)/lib -lutf8proc

all: $(BUILD)/clips

include mk/coreml.mk mk/tokenizer.mk mk/audio.mk mk/pipeline.mk mk/main.mk

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(CFLAGS_$*) -c -o $@ $<

check: $(CHECKS)

clean:
	rm -rf $(BUILD)

.PHONY: all check clean $(CHECKS)
