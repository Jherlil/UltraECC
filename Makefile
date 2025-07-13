.PHONY: default clean build bench fmt add mul rnd blf remote

CC = cc
NASM = nasm
CC_FLAGS ?= -O3 -funroll-loops -fomit-frame-pointer -ffast-math -Wall -Wextra -std=c11

ifeq ($(shell uname -m),x86_64)
        CC_FLAGS += -march=native -mavx2 -mbmi -mbmi2 -madx -msha -pthread -lpthread
endif

default: build

clean:
	@rm -rf ecloop bench main a.out *.profraw *.profdata xoshiro256ss-avx/*.o

build:
	$(MAKE) clean
	$(MAKE) xoshiro256ss-avx/xoshiro256ss.o
	$(CC) $(CC_FLAGS) -DXOSHIRO256SS_TECH=1 -I./xoshiro256ss-avx \
		main.c lib/flo-shani.c xoshiro256ss-avx/xoshiro256ss.c \
		xoshiro256ss-avx/xoshiro256ss.o -o ecloop

xoshiro256ss-avx/xoshiro256ss.o: xoshiro256ss-avx/xoshiro256ss.s
	$(NASM) -Ox -felf64 -DXOSHIRO256SS_TECH=1 -o $@ $<

bench: build
	./ecloop bench

fmt:
	@find . -name '*.c' | xargs clang-format -i
