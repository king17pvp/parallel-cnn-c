CC = gcc
CFLAGS = -Iinclude -O2
LDFLAGS = -lm

SRC = src/run_normal.c src/cnn.c src/utils.c
OBJ = $(SRC:.c=.o)

TARGET = build/cnn_exec_normal

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)


