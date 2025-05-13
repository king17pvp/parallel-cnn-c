CC = gcc
CFLAGS = -Iinclude -O2
LDFLAGS = -lm

SRCS = $(wildcard src/*.c)
TARGET = build/cnn_exec

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
