#ifndef MCG_OPTION_TYPES
#define MCG_OPTION_TYPES

struct MCGOptionTypes
{
  enum eSolver
  {
    BiCGStab,
    Gmres
  };

  enum ePreconditioner
  {
    NonePC,
    ILU0PC,
    PolyPC,
    FixpILU0PC,
    ColorILU0PC,
    BlockJacobiPC,
    BlockILU0PC,
    AMGX,
    HypreAMG
  };

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

#endif /* GPU_OPTION_TYPES */
