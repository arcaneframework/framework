


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
      message(status "IntelDPCPP FOUND : ${IntelDPCPP_FOUND}")
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
     
ENDIF()

if (TARGET intel)
    linkLibraries(alien_bench.exe intel)
endif ()

linkLibraries(alien_bench.exe
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_arcane_tools
        arcane_full
        )

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

commit(alien_bench.exe)

install(
        TARGETS alien_bench.exe
        RUNTIME DESTINATION bin
)

if(ALIEN_USE_SYCL)
if (TARGET hypre)
if (TARGET CUDA::cudart)
createExecutable(alien_hypre_sycl.exe)

addSources(alien_hypre_sycl.exe
           test_hypre_sycl.cpp
           hypre_sycl.cpp
          )

if(ALIEN_USE_SYCL)
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
     
ENDIF()

linkLibraries(alien_hypre_sycl.exe mpi)

linkLibraries(alien_hypre_sycl.exe 
              Boost::program_options)

linkLibraries(alien_hypre_sycl.exe hypre)

linkLibraries(alien_hypre_sycl.exe CUDA::cudart)

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

list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench.config
            AlienBench-aliensolver.arc
            )

if (TARGET petsc)
    configure_file(AlienBench.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench.arc COPYONLY)
    configure_file(AlienBench-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench.arc
            AlienBench-4P.arc
            AlienBench-petsc-spai.arc
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
    configure_file(AlienBench-trilinosmuelu.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinosmuelu.arc COPYONLY)
    configure_file(AlienBench-trilinosmuelu-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-trilinosmuelu-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench-trilinos.arc
            AlienBench-trilinos-2P.arc
            AlienBench-trilinosmuelu.arc
            AlienBench-trilinosmuelu-4P.arc
            )
endif ()
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
            BENCH alien.bench.aliensolver
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-aliensolver.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

if (TARGET petsc)

    alien_test(
            BENCH alien.bench
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    if(TARGET petsc::spai)
        alien_test(
                BENCH alien.bench.petsc-spai
                NAME seq
                PROCS 1
                COMMAND alien_bench.exe
                OPTIONS AlienBench-petsc-spai.arc
                WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

        alien_test(
                BENCH alien.bench.petsc-spai
                NAME parallel
                PROCS 4
                COMMAND alien_bench.exe
                OPTIONS AlienBench-petsc-spai-4P.arc
                WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
    endif()

endif ()

if (TARGET hypre)

    alien_test(
            BENCH alien.bench.hypre
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypre.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.hypre
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypre-2P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
            
if(ALIEN_USE_SYCL)
    alien_test(
            BENCH alien.bench.hypre.ilu
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypreilu.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
     
    alien_test(
            BENCH alien.bench.hypre.fsai
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hyprefsai.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

  if(TARGET CUDA::cudart)
    alien_test(
            BENCH alien.bench.hypre.ilu.sycl
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hypreilu-sycl.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
     
    alien_test(
            BENCH alien.bench.hypre.fsai.sycl
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-hyprefsai-sycl.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
  endif()
endif(ALIEN_USE_SYCL)

endif ()

if (TARGET trilinos)

    alien_test(
            BENCH alien.bench.trilinos
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.trilinosmuelu
            NAME seq
            PROCS 1
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinosmuelu.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.trilinos
            NAME parallel
            PROCS 2
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinos-2P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)

    alien_test(
            BENCH alien.bench.trilinosmuelu
            NAME parallel
            PROCS 4
            COMMAND alien_bench.exe
            OPTIONS AlienBench-trilinosmuelu-4P.arc
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench)
endif ()
