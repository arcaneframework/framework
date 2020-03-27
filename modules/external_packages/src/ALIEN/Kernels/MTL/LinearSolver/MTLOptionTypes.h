#ifndef MTL_OPTION_TYPES
#define MTL_OPTION_TYPES

struct MTLOptionTypes
{
  enum eSolver
  {
    GMRES,
    BiCGStab,
    CG,
    QR,
    LU
  };

  enum ePreconditioner
  {
    NonePC,
    DiagPC,
    ILU0PC,
    ILUTPC,
    SSORPC
  };
};

#endif /* MTL_OPTION_TYPES */
