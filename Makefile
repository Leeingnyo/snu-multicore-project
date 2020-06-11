TARGET=main
OBJECTS=util.o pix2pix.o

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3 -mavx2
# If you use MPI, use the following lines instead of above lines
#CXX=mpic++
#CXXFLAGS=-std=c++11 -Wall -O3 -DUSE_MPI

all: $(TARGET)

pix2pix.o: pix2pix.cpp pix2pix.h util.h
	g++ -c pix2pix.cpp -o pix2pix.o -std=c++11 -Wall -O3 -mavx

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
