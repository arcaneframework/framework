#ifndef TRILINOS_OPTION_TYPES
#define TRILINOS_OPTION_TYPES

struct TrilinosOptionTypes
{
  enum eSolver
  {
    BiCGStab,
    CG,
    GMRES,
    ML,
    MueLu,
    KLU2,
    NumOfSolver
  };
  static const std::string solver_type[NumOfSolver];
  static std::string const& solverName(eSolver solver) { return solver_type[solver]; }

  enum ePreconditioner
  {
    None,
    Relaxation,
    Chebyshev,
    ILUK,
    ILUT,
    FILU,
    Schwarz,
    MLPC,
    MueLuPC,
    NumOfPrecond
  };
  static const std::string preconditioner_type[NumOfPrecond];

  static std::string const& precondName(ePreconditioner precond)
  {
    return preconditioner_type[precond];
  }
};

#endif /* TRILINOS_OPTION_TYPES */
