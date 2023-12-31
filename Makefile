all: lamont

lamont: Matrix.h Matrix.cpp NeuralNetwork.h NeuralNetwork.cpp Main.cpp
	g++ -c Matrix.cpp
	g++ -c NeuralNetwork.cpp
	g++ -c Main.cpp
	g++ Matrix.o NeuralNetwork.o Main.o -o binary

run: lamont
	./binary

clean:
	rm -rf *.o *.gch binary
