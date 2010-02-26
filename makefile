CFLAGS:=$(shell pkg-config --cflags opencv) 
LDLIBS:=$(shell pkg-config --libs opencv)

sgmt: sgmt.o
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@
