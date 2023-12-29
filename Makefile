all: lamont

lamont: matrix.h matrix.cpp main.cpp
	g++ -c matrix.cpp
	g++ -c main.cpp
	g++ matrix.o main.o -o binary

run: lamont
	./binary

clean:
	rm -rf *.o *.gch binary
