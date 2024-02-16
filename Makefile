ifeq ($(shell uname), Darwin)
	CC = clang
	LEAK_CHECK = leaks --atExit --
else
	CC = gcc
	LEAK_CHECK = valgrind
endif

CFLAGS = -Wall -Wextra -std=c11
LDFLAGS = -lm
TARGET = main
OBJ = main.o

all: $(TARGET)

$(OBJ): main.c ml.h
	$(CC) $(CFLAGS) -c -o $(OBJ) main.c

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

leaks: $(TARGET)
	$(LEAK_CHECK) ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)
