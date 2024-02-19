ifeq ($(shell uname), Darwin)
	CC = clang
	LEAK_CHECK = leaks --atExit --
else
	CC = gcc
	LEAK_CHECK = valgrind
endif

# CFLAGS = -O2 -Wall -Wextra -std=c11
CFLAGS = -O2 -std=c11
LDFLAGS = -lm -lraylib -lGL -lpthread -ldl -lrt -lX11
TARGET = main
OBJ = main.o
TRAIN_OBJ = train.o
TRAIN_TARGET = train

all: $(TARGET)

$(OBJ): main.c ml.h
	$(CC) $(CFLAGS) -c -o $(OBJ) main.c

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(LDFLAGS)

$(TRAIN_OBJ): train.c ml.h
	$(CC) $(CFLAGS) -c -o $(TRAIN_OBJ) train.c

train: $(TRAIN_OBJ)
	$(CC) -o $(TRAIN_TARGET) $(TRAIN_OBJ) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

leaks: $(TARGET)
	$(LEAK_CHECK) ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ)
	rm -f $(TRAIN_TARGET) $(TRAIN_OBJ)
