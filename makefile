CFLAGS:=$(shell pkg-config --cflags opencv) -std=gnu99 -g
LDLIBS:=$(shell pkg-config --libs opencv)

sgmt: sgmt.o
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

clean:
	$(RM) sgmt sgmt.o
