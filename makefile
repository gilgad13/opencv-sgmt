CFLAGS:=$(shell pkg-config --cflags opencv) -std=gnu99
LDLIBS:=$(shell pkg-config --libs opencv)

sgmt: sgmt.o
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@
