PROGRAM		= tst-xoshiro256ss
STALIB		= ../libxoshiro256ss.a

SRCS		= tst-xoshiro256ss.c xoshiro256starstar_orig.c
OBJS		= $(SRCS:%=%.o)
DEPS		= $(SRCS:%=%.d)
INC		= ..

.PHONY: all build 
.DEFAULT_GOAL: all

CFLAGS		+= -DDEBUG -DTEST -g -Og -Wpedantic
ASMFLAGS	+= -DDEBUG -DTEST -g -Fdwarf

-include $(DEPS)

all: build

build: $(PROGRAM)

$(PROGRAM): $(OBJS) $(STALIB)
	$(CC) -o $@ $^ $(STALIB) $(LDLIBS) $(LDFLAGS)

%.c.o: %.c
	$(CC) -MMD -MP $(CFLAGS) -o $@ -c $< -I$(INC)

%.s.o: %.s
	$(ASM) $(ASMFLAGS) -o $@ $<


.PHONY: test
test:
	./$(PROGRAM)


.PHONY: clean dist-clean
clean:
	$(RM) *.o *.d
	$(RM) $(PROGRAM)

