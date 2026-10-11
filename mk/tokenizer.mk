# Stage 2 owns this fragment (see the Makefile header).
OBJS += $(BUILD)/tokenizer.o
CFLAGS_tokenizer = -I$(UTF8PROC)/include
CHECKS += check-tokenizer

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(BUILD)/tokenizer.o $(BUILD)/uni.o src/tokenizer.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer.c $(BUILD)/tokenizer.o $(BUILD)/uni.o $(LIBS)

check-tokenizer: $(BUILD)/test_tokenizer
	$(BUILD)/test_tokenizer
