#ifndef MCG_OPTION_TYPES
#define MCG_OPTION_TYPES

struct MCGOptionTypes
{
  enum eKernelType
  {
    CPU_CBLAS_BCSR,
    CPU_AVX_BCSR,
    CPU_AVX2_BCSP,
    CPU_AVX512_BCSP,
    GPU_CUBLAS_BELL,
    GPU_CUBLAS_BCSP
  };
};

#endif /* MCG_OPTION_TYPES */
