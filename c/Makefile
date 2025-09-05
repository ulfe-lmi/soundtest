.RECIPEPREFIX = >

CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra

all: encoder decoder

encoder: encoder.o dct.o entropy.o wav.o
>$(CC) $(CFLAGS) -o encoder encoder.o dct.o entropy.o wav.o -lm

decoder: decoder.o dct.o wav.o
>$(CC) $(CFLAGS) -o decoder decoder.o dct.o wav.o -lm

clean:
>rm -f encoder decoder *.o

.PHONY: all clean
