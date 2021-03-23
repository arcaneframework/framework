#ifndef HYPRE_OPTION_TYPES
#define HYPRE_OPTION_TYPES

struct HypreOptionTypes 
{
  enum eSolver
    {
      AMG,
      GMRES,
      BiCGStab,
      Hybrid
    };

  enum ePreconditioner
    {
      NoPC,
      AMGPC,
      ParaSailsPC,
      EuclidPC
    };
};

#endif /* HYPRE_OPTION_TYPES */
