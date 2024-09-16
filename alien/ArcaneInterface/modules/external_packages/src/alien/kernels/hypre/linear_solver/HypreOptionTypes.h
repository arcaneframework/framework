#ifndef HYPRE_OPTION_TYPES
#define HYPRE_OPTION_TYPES

struct HypreOptionTypes
{
  enum eSolver
  {
    AMG,
    CG,
    GMRES,
    BiCGStab,
    Hybrid
  };

  enum ePreconditioner
  {
    NoPC,
    DiagPC,
    AMGPC,
    ParaSailsPC,
    EuclidPC,
    ILUPC,
    FSAIPC
  };
};

#endif /* HYPRE_OPTION_TYPES */
