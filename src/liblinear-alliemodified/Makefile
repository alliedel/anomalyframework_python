CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -D CV_OMP
#CFLAGS = -Wall -Wconversion -g -fPIC -fopenmp -D CV_OMP
# for paralle CV and parameter search, add -D CV_OMP 
LIBS = blas/blas.a
SHVER = 3
OS = $(shell uname)
#LIBS = -lblas

all: train predict allie changenumbers manipulatetrain

lib: linear.o tron.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) -fopenmp $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: tron.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

allie: tron.o linear.o allie.c blas/blas.a
	$(CXX) $(CFLAGS) -o allie allie.c tron.o linear.o $(LIBS)

changenumbers: changenumbers.c
	$(CXX) $(CFLAGS) -o changenumbers changenumbers.c

manipulatetrain: tron.o linear.o manipulatetrain.c blas/blas.a
	$(CXX) $(CFLAGS) -g -O0 -o manipulatetrain manipulatetrain.c tron.o linear.o $(LIBS)

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -Wno-unused-result -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	make -C matlab clean
	rm -f *~ tron.o linear.o train predict liblinear.so.$(SHVER)
