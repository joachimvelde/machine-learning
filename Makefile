all:
	g++ -c matrix.cpp
	g++ -c main.cpp
	g++ matrix.o main.o -o binary

run: binary
	./binary

clean:
	rm -rf *.o *.gch binary
