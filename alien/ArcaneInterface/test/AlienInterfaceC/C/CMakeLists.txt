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


createExecutable(alien_interface_c.exe)

set_target_properties(alien_interface_c.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_c.exe
           main.c
           )
           

if(ALIEN_USE_SYCL)
    target_compile_definitions(alien_interface_c.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)

    IF(ALIEN_USE_INTELSYCL)
        set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
        IF(ALIEN_USE_HIP)
          set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
          target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
        ENDIF()
        IF(ALIEN_USE_CUDA)
          set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
          target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
          target_link_options(alien_interface_c.exe PRIVATE ${DPCPP_FLAGS})
        ENDIF()
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      #IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(alien_interface_c.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      #ENDIF()
      IF(ALIEN_USE_HIP)
        set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
        target_compile_options(alien_interface_c.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
      ENDIF()
       IF(ALIEN_USE_CUDA)
         #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         #target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
         #target_link_options(alien_bench.exe PRIVATE ${ACPP_FLAGS})
       ENDIF()
       target_compile_definitions(alien_interface_c.exe PRIVATE USE_ACPPSYCL)
    ENDIF()
ENDIF()
  
linkLibraries(alien_interface_c.exe
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        )
        
if (TARGET alien_core_solvers)
    linkLibraries(alien_interface_c.exe alien_core_solvers)
endif ()

if (TARGET alien_external_packages)
    linkLibraries(alien_interface_c.exe alien_external_packages)
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_c.exe alien_ifpen_solvers)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_c.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_c.exe alien_hpddm)
endif ()

commit(alien_interface_c.exe)

createExecutable(alien_interface_with_param_c.exe)

set_target_properties(alien_interface_with_param_c.exe PROPERTIES LINKER_LANGUAGE CXX)

addSources(alien_interface_with_param_c.exe
           main_with_parameters.c
           )
           

linkLibraries(alien_interface_with_param_c.exe
        Boost::boost
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_interface_c
        )

if (TARGET alien_core_solvers)
    linkLibraries(alien_interface_with_param_c.exe alien_core_solvers)
endif ()

if (TARGET alien_external_packages)
    linkLibraries(alien_interface_with_param_c.exe alien_external_packages)
endif ()

if (TARGET alien_ifpen_solvers)
    linkLibraries(alien_interface_with_param_c.exe alien_ifpen_solvers)
endif ()

if (TARGET trilinos)
    linkLibraries(alien_interface_with_param_c.exe alien_trilinos)
endif ()

if (TARGET hpddm)
    linkLibraries(alien_interface_with_param_c.exe alien_hpddm)
endif ()

commit(alien_interface_with_param_c.exe)

install(
        TARGETS alien_interface_c.exe alien_interface_with_param_c.exe
        RUNTIME DESTINATION bin
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c
            NAME seq
            PROCS 1
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c
            NAME parallel
            PROCS 4
            COMMAND alien_interface_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c.param
            NAME seq
            PROCS 1
            COMMAND alien_interface_with_param_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)

alien_test( BENCH interface.c.param
            NAME parallel
            PROCS 4
            COMMAND alien_interface_with_param_c.exe
            WORKING_DIRECTORY ${ALIEN_ARCANE_INTERFACE_DIR}/test/AlienInterfaceC/C)
