
all:
	clang++ -I./include/ -I/home/ts20/riscv-tool-chain/sysroot_openmp/include/ -c -o shmem_lib.o src/shmem_decoupling.cpp
	clang++ --target=riscv64-unknown-linux-gnu -mabi=lp64d --sysroot=/home/ts20/riscv-tool-chain/sysroot_riscv/riscv64-unknown-linux-gnu/riscv64-unknown-linux-gnu/sysroot/ --gcc-toolchain=/home/ts20/riscv-tool-chain/sysroot_riscv/riscv64-unknown-linux-gnu/ -I./include -c -o maple_lib.o src/maple_decoupling.cpp

clean:
	rm -f *~ shmem_lib.o maple_lib.o
