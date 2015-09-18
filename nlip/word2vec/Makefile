CC=gcc
CFLAGS=-shared -pthread -fPIC -fwrapv -Ofast -funroll-loops -Wall -fno-strict-aliasing -I/usr/include/python3.4

all:
	cython word2vec_fast.pyx
	$(CC) -o word2vec_fast.so word2vec_fast.c $(CFLAGS)

clean:
	rm word2vec_fast.so
