# Stage 4: discourse features + selection pipeline.
OBJS += $(BUILD)/features.o $(BUILD)/pipeline.o
CFLAGS_features = -I$(UTF8PROC)/include
CFLAGS_pipeline = -I$(UTF8PROC)/include
CHECKS += check-pipeline

$(BUILD)/test_pipeline: tests/test_pipeline.c $(BUILD)/features.o $(BUILD)/pipeline.o $(BUILD)/uni.o src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/test_pipeline.c $(BUILD)/features.o $(BUILD)/pipeline.o $(BUILD)/uni.o $(LIBS)

check-pipeline: $(BUILD)/test_pipeline
	$(BUILD)/test_pipeline
