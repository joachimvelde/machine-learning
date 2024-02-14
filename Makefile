ifeq ($(shell uname),Darwin)
	CC = clang
else
	CC = gcc
endif

CLFLAGS = -Wall -Wextra -std=c11
TARGET = main

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CLFANG) -o $(TARGET) main.c

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
