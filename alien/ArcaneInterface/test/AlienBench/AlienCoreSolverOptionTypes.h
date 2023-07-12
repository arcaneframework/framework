#pragma once

struct AlienCoreSolverOptionTypes
{

  enum eBackEnd
  {
    SimpleCSR,
    SYCL
  };

  enum eSolver
  {
    BCGS,
    CG
  };

  enum ePreconditioner
  {
    Diag,
    NeumannPoly,
    ChebyshevPoly,
    ILU0,
    FILU0,
  };
};

