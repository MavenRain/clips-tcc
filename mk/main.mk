# Stage 5: the clips binary and the end-to-end gate (runs build/clips: unsandboxed, Core ML cache).
CHECKS += check-e2e

$(BUILD)/clips: $(BUILD)/main.o $(OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(BUILD)/main.o $(OBJS) $(LIBS)

$(BUILD)/test_e2e: tests/test_e2e.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/test_e2e.c

check-e2e: fixtures/talk.mp4 $(BUILD)/clips $(BUILD)/test_e2e
	$(BUILD)/test_e2e
