#ifndef HPDDM_OPTION_TYPES
#define HPDDM_OPTION_TYPES

struct HPDDMOptionTypes
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

#endif /* HPDDM_OPTION_TYPES */
