CC = gcc
CFLAGS = -Iinclude -O2
LDFLAGS = -lm

SRC = src/main.c src/cnn.c src/utils.c
OBJ = $(SRC:.c=.o)

TARGET = cnn_exec

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)


