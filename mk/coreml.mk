# Stage 1: Core ML through the Objective-C runtime (src/coreml.c, tests/test_coreml.c).
# The test links only the objects declared before this point (uni.o, coreml.o).
OBJS += $(BUILD)/coreml.o
CHECKS += check-coreml

$(BUILD)/test_coreml: tests/test_coreml.c $(OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

check-coreml: $(BUILD)/test_coreml
	$(BUILD)/test_coreml
