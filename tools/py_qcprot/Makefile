OPT = -O3 -ffast-math 
WARN = -Wall -pedantic -std=c99
CFLAGS = $(WARN)
CC = gcc

OBJECTS = qcprot.o main.o

all:				testQCP

testQCP:			qcprot.o $(OBJECTS)
					$(CC) $(OPT) $(CFLAGS) $(OBJECTS) $(LIBS) -o testQCP

qcprot.o:			qcprot.c
					$(CC) $(OPT) $(CFLAGS) $(INCDIR) -c qcprot.c

main.o:				main.c
					$(CC) $(OPT) $(CFLAGS) $(INCDIR) -c main.c

clean:
					find . -name '*.o' -exec rm {} \;	
					rm testQCP &> /dev/null

