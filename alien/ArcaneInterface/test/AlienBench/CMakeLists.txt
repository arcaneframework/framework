


if(ALIEN_USE_SYCL)
    IF(ALIEN_USE_HIPSYCL)
      find_package(hipSYCL CONFIG REQUIRED)
      message(status "HIPSYCL                : ${hipSYCL_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()

    IF(ALIEN_USE_INTELDPCPP)
      find_package(IntelDPCPP CONFIG REQUIRED)
      message(status "IntelDPCPP : ${IntelDPCPP_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()
    IF(ALIEN_USE_INTELSYCL)
      find_package(IntelSYCL REQUIRED)
      message(status "IntelSYCL : ${IntelDPCPP_FOUND}")
      message(status "IntelSYCL FOUND : ${IntelDPCPP_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      find_package(AdaptiveCpp REQUIRED)
      message(status "AdaptiveSYCL           : ${AdaptiveCpp_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
      message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
      find_package(Boost COMPONENTS fiber context REQUIRED)
    ENDIF()
ENDIF()

createExecutable(alien_bench.exe)

addSources(alien_bench.exe
        main.cc
        AlienBenchModule.cc
        AlienStokesModule.cc
        ArcaneCodeService.cc
        )

generateAxl(alien_bench.exe
        AlienBench.axl
        AlienStokes.axl
        NO_COPY
        USER_INSTALL_PREFIX ALIEN
        )

if(ALIEN_USE_SYCL)
    addSources(alien_bench.exe
               AlienBenchSYCL.cc
              )
    target_compile_definitions(alien_bench.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
    IF(ALIEN_USE_HIPSYCL)
        add_sycl_to_target(TARGET  alien_bench.exe
                           SOURCES AlienBenchSYCL.cc)
    ENDIF()

    IF(ALIEN_USE_INTELSYCL)
        set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
        IF(ALIEN_USE_HIP)
          set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
          target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
        ENDIF()
        IF(ALIEN_USE_CUDA)
          set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
          target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
          target_link_options(alien_bench.exe PRIVATE ${DPCPP_FLAGS})
        ENDIF()
        add_sycl_to_target(TARGET  alien_bench.exe
                           SOURCES AlienBenchSYCL.cc)
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      #IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(alien_bench.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      #ENDIF()
      IF(ALIEN_USE_HIP)
        set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
        target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
      ENDIF()
       IF(ALIEN_USE_CUDA)
         #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         #target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
         #target_link_options(alien_bench.exe PRIVATE ${ACPP_FLAGS})
       ENDIF()
       target_compile_definitions(alien_bench.exe PRIVATE USE_ACPPSYCL)
       add_sycl_to_target(TARGET  alien_bench.exe
                           SOURCES AlienBenchSYCL.cc)
    ENDIF()

ENDIF()

if (TARGET intel)
    linkLibraries(alien_bench.exe intel)
endif ()

#        arcane_full
linkLibraries(alien_bench.exe
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_arcane_tools
        arcane_full
        )

if (TARGET  alien_core_solvers)
    linkLibraries(alien_bench.exe alien_core_solvers)
endif ()

if (TARGET  alien_external_packages)
    linkLibraries(alien_bench.exe alien_external_packages)
endif ()

if (TARGET  alien_ifpen_solvers)
    linkLibraries(alien_bench.exe alien_ifpen_solvers)
endif ()

if (TARGET  alien_trilinos)
    linkLibraries(alien_bench.exe alien_trilinos trilinos)
endif ()

if (TARGET alien_hpddm)
    linkLibraries(alien_bench.exe alien_hpddm)
endif ()

if (TARGET alien_composyx)
    linkLibraries(alien_bench.exe alien_composyx)
endif ()

commit(alien_bench.exe)

install(
        TARGETS alien_bench.exe
        RUNTIME DESTINATION bin
)

if(TARGET COMPOSYX::composyx)
  createExecutable(test_composyx.exe)
  addSources(test_composyx.exe test_composyx.cpp )
  linkLibraries(test_composyx.exe mpi COMPOSYX::composyx)
  commit(test_composyx.exe)
endif()

if (TARGET petsc)
if (TARGET CUDA::cudart)
createExecutable(test_petsc_cuda.exe)

addSources(test_petsc_cuda.exe test_petsc.cpp)
linkLibraries(test_petsc_cuda.exe mpi)

linkLibraries(test_petsc_cuda.exe
              Boost::program_options)

linkLibraries(test_petsc_cuda.exe petsc)
linkLibraries(test_petsc_cuda.exe CUDA::cudart)

commit(test_petsc_cuda.exe)
endif()
endif()

if(ALIEN_USE_SYCL)
if (TARGET hypre)
if (TARGET CUDA::cudart)
createExecutable(alien_hypre_sycl.exe)

addSources(alien_hypre_sycl.exe
           test_hypre_sycl.cpp
           hypre_sycl.cpp
          )

if(ALIEN_USE_SYCL)
    target_compile_definitions(alien_hypre_sycl.exe PUBLIC USE_SYCL2020)
    IF(ALIEN_USE_HIPSYCL)
        add_sycl_to_target(TARGET  alien_hypre_sycl.exe
                           SOURCES hypre_sycl.cpp)
    ENDIF()

    IF(ALIEN_USE_INTELSYCL)
        set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
        IF(ALIEN_USE_HIP)
          set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
          target_compile_options(alien_hypre_sycl.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
        ENDIF()
        IF(ALIEN_USE_CUDA)
          set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
          target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
          target_link_options(alien_hypre_sycl.exe PRIVATE ${DPCPP_FLAGS})
        ENDIF()
        add_sycl_to_target(TARGET  alien_hypre_sycl.exe
                           SOURCES hypre_sycl.cpp)
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(alien_hypre_sycl.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      IF(ALIEN_USE_HIP)
        set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
        target_compile_options(alien_hypre_sycl.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
      ENDIF()
      IF(ALIEN_USE_CUDA)
         #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         #target_compile_options(alien_hypre_sycl.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
         #target_link_options(alien_hypre_sycl.exe PRIVATE ${ACPP_FLAGS})
      ENDIF()
      target_compile_definitions(alien_hypre_sycl.exe PRIVATE USE_ACPPSYCL)
      add_sycl_to_target(TARGET  alien_hypre_sycl.exe
                           SOURCES hypre_sycl.cpp)
    ENDIF()

ENDIF()

linkLibraries(alien_hypre_sycl.exe mpi)

linkLibraries(alien_hypre_sycl.exe
              Boost::program_options)

linkLibraries(alien_hypre_sycl.exe hypre)
if (TARGET  alien_external_packages)
    linkLibraries(alien_hypre_sycl.exe alien_external_packages)
endif ()

if (TARGET intel)
    linkLibraries(alien_hypre_sycl.exe intel)
endif ()

if(TARGET CUDA::cudart)
linkLibraries(alien_hypre_sycl.exe CUDA::cudart)
endif()

if(ALIEN_USE_SYCL)
    IF(ALIEN_USE_ACPPSYCL)
        linkLibraries(alien_hypre_sycl.exe
                      Boost::fiber Boost::context)
    ENDIF()
ENDIF()

commit(alien_hypre_sycl.exe)

install(
        TARGETS alien_hypre_sycl.exe
        RUNTIME DESTINATION bin
)
endif ()
endif ()
endif ()

configure_file(AlienBench.config ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench.config COPYONLY)
configure_file(AlienBench-aliensolver.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver.arc COPYONLY)
configure_file(AlienBench-aliensolver-ilu0.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-ilu0.arc COPYONLY)
configure_file(AlienBench-aliensolver-ilu0-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-ilu0-2P.arc COPYONLY)
configure_file(AlienBench-aliensolver-ilu0-bs2.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-ilu0-bs2.arc COPYONLY)
configure_file(AlienBench-aliensolver-filu0.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-filu0.arc COPYONLY)
configure_file(AlienBench-aliensolver-cheb.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-cheb.arc COPYONLY)
configure_file(AlienBench-aliensolver-cheb-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-cheb-2P.arc COPYONLY)
configure_file(AlienBench-aliensolver-cheb-bs2.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-cheb-bs2.arc COPYONLY)
configure_file(AlienBench-aliensolver-amg.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-amg.arc COPYONLY)
configure_file(AlienBench-aliensolver-cxramg-bs2.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-aliensolver-cxramg-bs2.arc COPYONLY)

list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench.config
            AlienBench-aliensolver.arc
            AlienBench-aliensolver-ilu0.arc
            AlienBench-aliensolver-ilu0-2P.arc
            AlienBench-aliensolver-ilu0-bs2.arc
            AlienBench-aliensolver-filu0.arc
            AlienBench-aliensolver-cheb.arc
            AlienBench-aliensolver-cheb-2P.arc
            AlienBench-aliensolver-cheb-bs2.arc
            AlienBench-aliensolver-amg.arc
            AlienBench-aliensolver-cxramg.arc
            )

if (TARGET petsc)
    configure_file(AlienBench.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench.arc COPYONLY)
    configure_file(AlienBench-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-4P.arc COPYONLY)
    configure_file(AlienBench-petsc-gamg.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-petsc-gamg.arc COPYONLY)
    configure_file(AlienBench-petsc-gamg-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-petsc-gamg-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench.arc
            AlienBench-4P.arc
            AlienBench-petsc-gamg.arc
            AlienBench-petsc-gamg-4P.arc
            )

    if (TARGET petsc::spai)
        configure_file(AlienBench-petsc-spai.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-petsc-spai.arc COPYONLY)
        configure_file(AlienBench-petsc-spai-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-petsc-spai-4P.arc COPYONLY)

        list(APPEND ALIEN_EXAMPLE_FILES
                AlienBench-petsc-spai.arc
                AlienBench-petsc-spai-4P.arc)
    endif ()
endif ()

if (TARGET hypre)
    configure_file(AlienBench-hypre.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hypre.arc COPYONLY)
    configure_file(AlienBench-hypre-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hypre-2P.arc COPYONLY)
    configure_file(AlienBench-hypreilu.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hypreilu.arc COPYONLY)
    configure_file(AlienBench-hypreilu-sycl.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hypreilu-sycl.arc COPYONLY)
    configure_file(AlienBench-hyprefsai.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hyprefsai.arc COPYONLY)
    configure_file(AlienBench-hyprefsai-sycl.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hyprefsai-sycl.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-hypre.arc
            AlienBench-hypre-2P.arc
            )
if(ALIEN_USE_SYCL)
  if(TARGET CUDA::cudart)
    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-hypreilu.arc
            AlienBench-hyprefsai.arc
            AlienBench-hypreilu-sycl.arc
            AlienBench-hyprefsai-sycl.arc)
  endif()
endif(ALIEN_USE_SYCL)
endif ()

if (TARGET trilinos)
    configure_file(AlienBench-trilinos.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinos.arc COPYONLY)
    configure_file(AlienBench-trilinos-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinos-2P.arc COPYONLY)
    configure_file(AlienBench-trilinos-bs2.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinos-bs2.arc COPYONLY)
    configure_file(AlienBench-trilinos-bs2-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinos-bs2-4P.arc COPYONLY)
    configure_file(AlienBench-trilinosmuelu.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinosmuelu.arc COPYONLY)
    configure_file(AlienBench-trilinosmuelu-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinosmuelu-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-trilinos.arc
            AlienBench-trilinos-2P.arc
            AlienBench-trilinos-bs2.arc
            AlienBench-trilinos-bs2-4P.arc
            AlienBench-trilinosmuelu.arc
            AlienBench-trilinosmuelu-4P.arc
            )
endif ()

if (TARGET ifpsolver)
    configure_file(AlienBench-ifps.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-ifps.arc COPYONLY)
    configure_file(AlienBench-ifps-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-ifps-2P.arc COPYONLY)
    configure_file(AlienBench-ifpsamg.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-ifpsamg.arc COPYONLY)
    configure_file(AlienBench-ifpsamg-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-ifpsamg-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-ifps.arc
            AlienBench-ifps-2P.arc
            AlienBench-ifpsamg.arc
            AlienBench-ifpsamg-4P.arc
            )
endif ()

if (TARGET htssolver)
    configure_file(AlienBench-hts.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hts.arc COPYONLY)
    configure_file(AlienBench-hts-2P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-hts-2P.arc COPYONLY)
    configure_file(AlienBench-htsamg.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-htsamg.arc COPYONLY)
    configure_file(AlienBench-htsamg-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-htsamg-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-hts.arc
            AlienBench-hts-2P.arc
            AlienBench-htsamg.arc
            AlienBench-htsamg-4P.arc
            )
endif ()

if(TARGET COMPOSYX::composyx)

    configure_file(AlienBench-composyx.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx.arc COPYONLY)
    configure_file(AlienBench-composyx-gmres.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx-gmres.arc COPYONLY)
    configure_file(AlienBench-composyx-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx-4P.arc COPYONLY)
    configure_file(AlienBench-composyx-bcgsas-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx-bcgsas-4P.arc COPYONLY)
    configure_file(AlienBench-composyx-gmres-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx-gmres-4P.arc COPYONLY)
    configure_file(AlienBench-composyx-gmresas-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-composyx-gmresas-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES AlienBench-composyx.arc
                                    AlienBench-composyx-gmres.arc
                                    AlienBench-composyx-4P.arc
                                    AlienBench-composyx-gmres-4P.arc)

endif()

#Examples !

get_property(ALIEN_COMPILE_DEFS SOURCE main.cc PROPERTY COMPILE_DEFINITIONS)
set(ALIEN_EXAMPLE_NAME "AlienExample.exe")

configure_file(Makefile.in Makefile.example @ONLY)

install(FILES
        ${CMAKE_BINARY_DIR}/ALIEN/axl/AlienBench_axl.h
        AlienBenchModule.h
        AlienBenchModule.cc
        main.cc
        DESTINATION share/examples
        )

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Makefile.example
        DESTINATION share/examples
        RENAME Makefile
        )

alien_test(
            BENCH bench.aliensolver
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

alien_test(
            BENCH bench.aliensolver-ilu0
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-ilu0.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if(TARGET eigen3)
alien_test(
            BENCH bench.aliensolver-ilu0-b2
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-ilu0-bs2.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif()

alien_test(
            BENCH bench.aliensolver-ilu0
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-ilu0-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

alien_test(
            BENCH bench.aliensolver-cheb
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-cheb.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if(TARGET eigen3)
alien_test(
            BENCH bench.aliensolver-cheb-bs2
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-cheb-bs2.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif()

alien_test(
            BENCH bench.aliensolver-cheb
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-cheb-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if(TARGET hypre)
alien_test(
            BENCH bench.aliensolver-amg
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-amg.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if(TARGET eigen3)
alien_test(
            BENCH bench.aliensolver-cxramg
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver-cxramg-bs2.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif()
endif()

if (TARGET petsc)

    alien_test(
            BENCH bench
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.petsc-gamg
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-petsc-gamg.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.petsc.gamg
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-petsc-gamg-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    if(TARGET petsc::spai)
        alien_test(
                BENCH bench.petsc-spai
                NAME seq
                PROCS 1
                COMMAND alien_bench.exe
                OPTIONS AlienBench-petsc-spai.arc
                UNIQUE_OUTPUT_DIR
                WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

        alien_test(
                BENCH bench.petsc-spai
                NAME parallel
                PROCS 4
                COMMAND alien_bench.exe
                OPTIONS AlienBench-petsc-spai-4P.arc
                UNIQUE_OUTPUT_DIR
                WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
    endif()

endif ()

if (TARGET hypre)

    alien_test(
            BENCH bench.hypre
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypre.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.hypre
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypre-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if(ALIEN_USE_SYCL)
    alien_test(
            BENCH bench.hypre.ilu
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypreilu.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.hypre.fsai
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hyprefsai.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

  if(TARGET CUDA::cudart)
    alien_test(
            BENCH bench.hypre.ilu.sycl
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypreilu-sycl.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.hypre.fsai.sycl
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hyprefsai-sycl.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
  endif()
endif(ALIEN_USE_SYCL)

endif ()

if (TARGET trilinos)

    alien_test(
            BENCH bench.trilinos
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.trilinos-bs2
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos-bs2.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.trilinosmuelu
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinosmuelu.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.trilinos
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.trilinos-bs2
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos-bs2-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.trilinosmuelu
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinosmuelu-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif ()


if (TARGET ifpsolver)

    alien_test(
            BENCH bench.ifps
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-ifps.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.ifpsamg
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-ifpsamg.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.ifps
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-ifps-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.ifpsamg
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-ifpsamg-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif ()

if (TARGET htssolver)

    alien_test(
            BENCH bench.hts
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hts.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.htsamg
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-htsamg.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.hts
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hts-2P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH bench.htsamg
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-htsamg-4P.arc
            UNIQUE_OUTPUT_DIR
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif ()

if (TARGET COMPOSYX::composyx)

    alien_test(
            BENCH alien.bench.composyx.bcgs
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.composyx.gmres
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx-gmres.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.composyx.bcgs
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.composyx.bcgs.as
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx-bcgsas-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.composyx.gmres
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx-gmres-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.composyx.gmres.as
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-composyx-gmresas-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

endif()
