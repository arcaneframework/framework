#include <gtest/gtest.h>

#include <ALIEN/Alien-RefSemanticMVHandlers.h>
#include <ALIEN/AlienExternalPackages.h>

#ifdef ALIEN_USE_HYPRE
#include <ALIEN/Kernels/Hypre/linear_solver/HypreInternalLinearSolver.h>
#include <ALIEN/Kernels/Hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/Kernels/Hypre/linear_solver/HypreOptionTypes.h>
#include <ALIEN/Kernels/Hypre/data_structure/HypreMatrix.h>
#include <ALIEN/axl/HypreSolver_IOptions.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>
#endif
#ifdef ALIEN_USE_IFPSOLVER
#include <ALIEN/Kernels/ifp/linear_solver/arcane/IFPLinearSolverService.h>
#include <ALIEN/Kernels/ifp/linear_solver/IFPSolverProperty.h>
#include <ALIEN/axl/IFPLinearSolver_IOptions.h>
#include <ALIEN/axl/IFPLinearSolver_StrongOptions.h>
#endif

namespace Environment {
extern Alien::IParallelMng* parallelMng();
}

TEST(TestHypreIFPS, HypreTest)
{
  using namespace Alien;
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, Environment::parallelMng());
  Alien::VectorDistribution vdist(s, Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Vector x(vdist); // x.setName("x") ;
  Alien::Vector y(vdist); // y.setName("y") ;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  auto tag = Alien::DirectMatrixOptions::eResetValues;
  {
    Alien::DirectMatrixBuilder builder(A, tag);
    builder.reserve(3 * local_size);
    builder.allocate();
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }
  {
    Alien::LocalVectorWriter writer(x);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 1.;
  }
  {
    Alien::LocalVectorWriter writer(y);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 0.;
  }

#ifdef ALIEN_USE_IFPSOLVER
  {
    using namespace IFPLinearSolverOptionsNames;
    std::shared_ptr<IOptionsIFPLinearSolver> options(
        new StrongOptionsIFPLinearSolver{ _output = 1, _numIterationsMax = 1000,
            _stopCriteriaValue = 1e-6, _precondOption = IFPSolverProperty::AMG });
    auto* solver =
        new Alien::IFPInternalLinearSolver(Environment::parallelMng(), options.get());
    //
    solver->init();
    // solver->solve(A,x,y);
    const Alien::SolverStatus& status = solver->getStatus();
    // solver->end();
  }
#endif
#ifdef ALIEN_USE_HYPRE
  {
    A.impl()->get<Alien::BackEnd::tag::hypre>();
    using namespace HypreSolverOptionsNames;
    // std::shared_ptr<IOptionsHypreSolver> options(new StrongOptionsHypreSolver());
    auto* solver =
        new Alien::HypreInternalLinearSolver(Environment::parallelMng(), nullptr);
    //
    solver->init();
    // solver->solve(A,y,x);
  }
#endif
}
