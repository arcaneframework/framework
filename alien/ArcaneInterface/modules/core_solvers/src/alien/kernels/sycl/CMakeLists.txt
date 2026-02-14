
if(ALIEN_USE_SYCL)
    addSources(alien_core_sycl_solvers
               linear_solver/AlienCoreSYCLLinearSolver.cc
               )

    target_compile_definitions(alien_core_sycl_solvers PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
    IF(ALIEN_USE_HIPSYCL)
        add_sycl_to_target(TARGET  alien_core_sycl_solvers
                           SOURCES linear_solver/AlienCoreSYCLLinearSolver.cc)
    ENDIF()

    IF(ALIEN_USE_INTELSYCL)
        set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
        IF(ALIEN_USE_HIP)
          set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
          target_compile_options(alien_core_sycl_solvers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
        ENDIF()
        IF(ALIEN_USE_CUDA)
          set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
          target_compile_options(alien_core_sycl_solvers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
          target_link_options(alien_core_sycl_solvers PRIVATE ${DPCPP_FLAGS})
        ENDIF()
        add_sycl_to_target(TARGET  alien_core_sycl_solvers
                           SOURCES linear_solver/AlienCoreSYCLLinearSolver.cc)
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      #IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(alien_core_sycl_solvers PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      #ENDIF()
      IF(ALIEN_USE_HIP)
        set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
        target_compile_options(alien_core_sycl_solvers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
      ENDIF()
       IF(ALIEN_USE_CUDA)
         #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         #target_compile_options(alien_bench.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
         #target_link_options(alien_bench.exe PRIVATE ${ACPP_FLAGS})
       ENDIF()
       target_compile_definitions(alien_core_sycl_solvers PRIVATE USE_ACPPSYCL SYCL_VENDOR_ADAPTIVECPP _SYCL_INTERNAL_ADAPTIVECPP_USE_CL_SYCL_NAMESPACE)
       add_sycl_to_target(TARGET  alien_core_sycl_solvers
                           SOURCES linear_solver/AlienCoreSYCLLinearSolver.cc)
    ENDIF()

ENDIF()
