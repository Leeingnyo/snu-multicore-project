TARGET=main
OBJECTS=util.o pix2pix.o

#CXX=g++
#CXXFLAGS=-std=c++11 -Wall -O3
# If you use MPI, use the following lines instead of above lines
CXX=mpic++
CXXFLAGS=-std=c++11 -Wall -O3 -DUSE_MPI -fopenmp -mavx
LDFLAGS=-lm -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	thorq --add --mode mpi --nodes 1 --slots 2 --timeout 3600 --device gpu/1080 ./$(TARGET) ../common/edges2cats_AtoB.bin $(ARGS)
