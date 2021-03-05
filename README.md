# The GraphAttack Compiler

The GraphAttack compiler is adapted from the [DECADES compiler](https://github.com/PrincetonUniversity/DECADEScompiler).
Below are instructions on how to build and run the compiler:

Prerequisites -----

DEC++ is built on LLVM/Clang. It is able to compile to riscv64
binaries using the riscv gcc sysroot.

To obtain these (large) depedencies, please go to the ext directory
and run build. This will a large install (~13 GB) and take several
hours. It will build LLVM and riscv64 GCC and OpenMP.

Building DEC++
-----

Once you have successfully installed/acquired the right versions of CMake and LLVM, run the following:

    mkdir build
    cd build
    cmake ../
    make

be sure that cmake is pointing to the right version of LLVM, this can be done with CMAKE variables, e.g.:

cmake -DLLVM_DIR=/home/ts20/DECADES_compiler/ext/build_llvm/lib/cmake/llvm ../

Running DEC++
-----

DEC++ can be invoked from the command line with the following possible arguments:

    DEC++ -m MODE [-t] [NUM_THREADS] [-sps] [SCRATCHPAD_SIZE] [-s] [BISCUIT_SYNC_MODE] [-spp] [SIMULATOR_PREPROCESSING_SCRIPT] [--target] [TARGET] $FILE [$ARGS]

`MODE` must be one of the following:

- Native ("n"): simply compiles the source files with no LLVM transformations or passes (useful for verification of correct program output and debugging, as well as application development outside of any DECADES features)

        DEC++ -m n $FILE [$ARGS]

    No additional arguments are necessary.

- DECADES Base ("db"): the default DEC++ mode that identifies the `kernel` function, performs function inlining and wraps the function invocation in the tile launcher

        DEC++ -m db [-t] [NUM_THREADS] [--target] [TARGET] $FILE [$ARGS]

    where `NUM_THREADS` is the number of threads to utilize in parallel and `TARGET` can be either "x86" (default), "simulator" (https://github.com/PrincetonUniversity/pythia), or "riscv64" (generate a RISC-V binary to run on the DECADES FPGA emulation and chip platform). If the target is "simulator", then the path to the simulator preprocessing script is necessary:

        DEC++ -m db [-t] [NUM_THREADS] [-spp] [SIMULATOR_PREPROCESSING_SCRIPT] [--target] [TARGET] $FILE [$ARGS]

- Decoupled Implicit ("di"): the decoupling compilation mode that slices the `kernel` function into supply and compute programs completely automatically

        DEC++ -m di [-t] [NUM_THREADS] [--target] [TARGET] $FILE [$ARGS]

    where `NUM_THREADS` is the number of threads to utilize in parallel and `TARGET` can be either "x86" (default) or "simulator".

## API Documentation

In order to integrate other core or accelerator models with MosaicSim and have them interract together, we have documented an API. See Section E in the linked document: https://www.overleaf.com/project/5c87bee2b8ed496eb059acfb

We also provide further documentation about the simulator and compiler within the DECADES project in the next repo:

  https://github.com/PrincetonUniversity/decades_documentation

## Workloads

We provide further workloads, used withing the DECADES project in the next repo:

  https://github.com/amanocha/DECADES_Applications

## License

  [BSD License (BSD 2-Clause License)](BSD-License.txt)
