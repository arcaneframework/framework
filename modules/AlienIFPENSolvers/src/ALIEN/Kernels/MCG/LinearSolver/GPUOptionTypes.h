#ifndef GPU_OPTION_TYPES
#define GPU_OPTION_TYPES

struct GPUOptionTypes
{
  enum eSolver
    {
      BiCGStab,
      Gmres
    };

  enum ePreconditioner
    {
      NonePC,
      DiagPC,
      ILU0PC,
      PolyPC,
	    PolyNKPC,
      BSSORPC,
      ColorBlockILU0PC,
      CprPC
    };

  enum eKernelType
  {
    GPUKernel,
    GPUNvidiaKernel,
    CPUKernel,
    CPUCBLASKernel,
    CPUSSE2Kernel,
    CPUAVXKernel,
    GPUCUBLASEllSpmvKernel,
    GPUCUBLASBSRSpmvKernel,
    GPUCUBLASIFPENV2SpmvKernel, // obsolete
    GPUCUBLASIFPENV3SpmvKernel, // obsolete
    GPUCUBLASHybridSpmvKernel,  // obsolete
    GPUCUBLASBELLSpmvKernel,
    GPUCUBLASBCSPSpmvKernel,
    MCKernel
  } ;
};

#endif /* GPU_OPTION_TYPES */
