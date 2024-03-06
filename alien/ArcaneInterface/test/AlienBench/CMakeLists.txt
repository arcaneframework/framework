


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
    linkLibraries(alien_bench.exe alien_trilinos)
endif ()

if (TARGET alien_hpddm)
    linkLibraries(alien_bench.exe alien_hpddm)
endif ()

commit(alien_bench.exe)

install(
        TARGETS alien_bench.exe
        RUNTIME DESTINATION bin
)

if (TARGET petsc)

    configure_file(AlienBench.config ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench.config COPYONLY)
    configure_file(AlienBench.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench.arc COPYONLY)
    configure_file(AlienBench-4P.arc ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienBench/AlienBench-4P.arc COPYONLY)

    list(APPEND ALIEN_EXAMPLE_FILES
            AlienBench.config
            AlienBench.arc
            AlienBench-4P.arc
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


if (TARGET petsc)

    set(ENV (ARCANE_PARALLEL_SERVICE) "Mpi")

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

endif ()
