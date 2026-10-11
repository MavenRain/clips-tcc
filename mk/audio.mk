# Stage 3: audio decode (libav from ffmpeg-full) and ASR words (whisper.cpp).
AUDIO_FFMPEG = /opt/homebrew/opt/ffmpeg-full
WORDS_WHISPER = /opt/homebrew/opt/whisper-cpp
WORDS_GGML = /opt/homebrew/opt/ggml

OBJS += $(BUILD)/audio.o $(BUILD)/words.o
CFLAGS_audio = -I$(AUDIO_FFMPEG)/include
CFLAGS_words = -I$(WORDS_WHISPER)/include -I$(WORDS_GGML)/include -I$(UTF8PROC)/include
LIBS += -L$(AUDIO_FFMPEG)/lib -lavformat -lavcodec -lswresample -lavutil
LIBS += -L$(WORDS_WHISPER)/lib -lwhisper -L$(WORDS_GGML)/lib -lggml
CHECKS += check-audio check-words

$(BUILD)/test_audio: tests/test_audio.c $(BUILD)/audio.o src/audio.h | $(BUILD)
	$(CC) $(CFLAGS) $(CFLAGS_audio) -DTEST_FFMPEG='"$(AUDIO_FFMPEG)/bin/ffmpeg"' -o $@ tests/test_audio.c $(BUILD)/audio.o $(LIBS)

check-audio: $(BUILD)/test_audio
	$(BUILD)/test_audio

$(BUILD)/test_words: tests/test_words.c $(BUILD)/audio.o $(BUILD)/words.o $(BUILD)/uni.o src/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(CFLAGS_audio) -o $@ tests/test_words.c $(BUILD)/audio.o $(BUILD)/words.o $(BUILD)/uni.o $(LIBS)

fixtures/talk.mp4: fixtures/talk.txt | $(BUILD)
	/usr/bin/say -v Samantha -f $< -o $(BUILD)/talk.aiff
	$(AUDIO_FFMPEG)/bin/ffmpeg -y -nostdin -loglevel error -f lavfi -i color=c=black:s=320x240:r=10 -i $(BUILD)/talk.aiff -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac $(BUILD)/talk.mp4
	mv $(BUILD)/talk.mp4 $@

check-words: fixtures/talk.mp4 $(BUILD)/test_words
	$(BUILD)/test_words
