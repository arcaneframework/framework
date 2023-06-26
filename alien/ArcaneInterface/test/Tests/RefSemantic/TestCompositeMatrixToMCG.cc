#include <gtest/gtest.h>

#include <alien/AlienExternalPackages.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/functional/Cast.h>
#include <alien/data/CompositeVector.h>
#include <alien/data/CompositeMatrix.h>

#ifdef ALIEN_USE_MCGSOLVER
#include <alien/kernels/mcg/linear_solver/arcane/MCGLinearSolver.h>
#include <alien/kernels/mcg/linear_solver/MCGOptionTypes.h>
#include <ALIEN/axl/MCGSolver_IOptions.h>
#include <ALIEN/axl/MCGSolver_StrongOptions.h>
#endif

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

auto mcgfill = [](Alien::IVector& c, Alien::Real shift) {
  const Alien::VectorDistribution& dist = c.impl()->distribution();
  Alien::Integer lsize = dist.localSize();
  Alien::Integer offset = dist.offset();
  std::cout << lsize << ", " << offset << std::endl;
  auto& cc = Alien::cast<Alien::Vector>(c);
  Alien::VectorWriter writer(cc);
  for (Alien::Integer i = 0; i < lsize; ++i)
    writer[i] = offset + i + shift;
};

#if 0
TEST(TestCompositeMatrixToMCG, CompositeMCGTest)
{
#ifdef ALIEN_USE_MCGSOLVER
  Alien::Integer size = 5;
  Alien::Integer nbSubMatrices = 2;
  Alien::MatrixDistribution dist(size, size, Environment::parallelMng());
  Alien::VectorDistribution vecDist(size, Environment::parallelMng());
  // Alien::CompositeMatrixData v(size*nbSubMatrices, nbSubMatrices); // matrice 10x10
  // Alien::CompositeVectorData vec(size*nbSubMatrices, nbSubMatrices); // vecteur 10x1
  // Alien::CompositeVectorData vecSol(size*nbSubMatrices, nbSubMatrices); // vecteur 10x1
  Alien::CompositeMatrix v;
  v.resize(nbSubMatrices);
  Alien::CompositeVector vec;
  vec.resize(nbSubMatrices);
  Alien::CompositeVector vecSol;
  vecSol.resize(nbSubMatrices);
  for (Alien::Integer i = 0; i < nbSubMatrices; ++i) {
    Alien::CompositeElement(vec, i) = Alien::Vector(vecDist);
    auto& vi = vec[i];
    ::mcgfill(vi, 1.);
    Alien::CompositeElement(vecSol, i) = Alien::Vector(vecDist);
    auto& vSol = vecSol[i];
    ::mcgfill(vSol, 0.);
    for (Alien::Integer j = 0; j < nbSubMatrices; ++j) {
      Alien::CompositeElement(v, i, j) = Alien::Matrix(dist);
      auto& cij = v(i, j);
      auto& M = Alien::cast<Alien::Matrix>(cij);
      auto tag = Alien::DirectMatrixOptions::eResetValues;
      Alien::DirectMatrixBuilder builder(M, tag);
      builder.reserve(size);
      builder.allocate();
      for (Alien::Integer i = 0; i < size; ++i) {
        builder(i, i) = 2;
        if (i > 0)
          builder(i, i - 1) = -0.1;
        if (i < size - 1)
          builder(i, i + 1) = -0.1;
      }
    }
  }

  ASSERT_TRUE(vec.hasUserFeature("composite"));
  ASSERT_EQ(nbSubMatrices, vec.size());
  ASSERT_EQ(nbSubMatrices * size, vec.space().size());
  ASSERT_TRUE(vecSol.hasUserFeature("composite"));
  ASSERT_EQ(nbSubMatrices, vecSol.size());
  ASSERT_EQ(nbSubMatrices * size, vecSol.space().size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ(nbSubMatrices, v.size());
  ASSERT_EQ(nbSubMatrices * size, v.rowSpace().size());
  ASSERT_EQ(nbSubMatrices * size, v.colSpace().size());
  //
  using namespace MCGSolverOptionsNames;
  std::shared_ptr<IOptionsMCGSolver> options(new StrongOptionsMCGSolver{ _output = 1,
      _maxIterationNum = 1000, _stopCriteriaValue = 1e-6,
      _kernel = MCGOptionTypes::eKernelType::CPU_CBLAS_BCSR,
      _solver = MCGSolver::BiCGS,
      _preconditioner = MCGSolver::PrecILU0 });
  auto* solver = new Alien::MCGLinearSolver(Environment::parallelMng(), options);
  //
  solver->init();
  solver->solve(v, vec, vecSol);
  const Alien::SolverStatus& status = solver->getStatus();
  if (status.succeeded) {
    std::cout << "Solver succeed   " << status.succeeded << std::endl;
    std::cout << "             residual " << status.residual << std::endl;
    std::cout << "             nb iters  " << status.iteration_count << std::endl;
    std::cout << "             error      " << status.error << std::endl;
  }
  delete solver;
#endif
}
#endif