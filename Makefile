ifeq ($(shell uname),Darwin)
	CC = clang
	LEAK_CHECK = leaks --atExit --
else
	CC = gcc
	LEAK_CHECK = valgrind
endif

CLFLAGS = -Wall -Wextra -std=c11
TARGET = main

all: $(TARGET)

$(TARGET): main.c ml.h
	$(CC) $(CLFANG) -o $(TARGET) main.c

run: $(TARGET)
	./$(TARGET)

leaks: $(TARGET)
	$(LEAK_CHECK) ./$(TARGET)

clean:
	rm -f $(TARGET)
