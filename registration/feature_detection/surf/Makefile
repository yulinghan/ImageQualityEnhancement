CCPP=g++
CC=gcc

CFLAGS=`pkg-config --cflags opencv4` 
LDFLAGS=`pkg-config --libs opencv4` -lpthread

SOURCES_CPP := $(shell find . -name '*.cpp')
SOURCES_C := $(shell find . -name '*.c')
OBJ := $(SOURCES_CPP:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
HEADERS := $(shell find . -name '*.h')

all: myBin

.cpp.o:  %.cpp %.h
	$(CCPP) -O2 -o $@ $(CFLAGS) -c $+

.c.o:  %.c %.h
	$(CC) -O2 -o $@ $(CFLAGS) -c $+

myBin: $(HEADERS) $(OBJ)
	$(CCPP) -O2 -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJ) myBin
