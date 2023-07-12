#ifndef HTS_OPTION_TYPES
#define HTS_OPTION_TYPES

struct HTSOptionTypes
{
  enum eSolver
  {
    BiCGStab,
    DDML
  };

  enum ePreconditioner
  {
    None,
    Poly,
    Chebyshev,
    BSSOR,
    ILU0,
    ILU0FP,
    Cpr,
    DDMLPC,
    AMGPC,
    CprAMG,
    CprDDML
  };
};

#endif /* HTS_OPTION_TYPES */
