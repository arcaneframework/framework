#include <gtest/gtest.h>

#include <ALIEN/AlienExternalPackages.h>
#include <ALIEN/Alien-RefSemantic.h>

#ifdef ALIEN_USE_MCGSOLVER
#include <ALIEN/Kernels/MCG/LinearSolver/Arcane/GPULinearSolver.h>
#include <ALIEN/Kernels/MCG/LinearSolver/GPUOptionTypes.h>
#include <ALIEN/axl/GPUSolver_IOptions.h>
#include <ALIEN/axl/GPUSolver_StrongOptions.h>
#endif

namespace Environment {
extern Arccore::MessagePassing::IMessagePassingMng* parallelMng();
}

auto fill = [&](Alien::IVector& c, Alien::Real shift) {
  const Alien::VectorDistribution& dist = c.impl()->distribution();
  Alien::Integer lsize = dist.localSize();
  Alien::Integer offset = dist.offset();
  std::cout << lsize << ", " << offset << std::endl;
  auto& cc = Alien::cast<Alien::Vector>(c);
  Alien::VectorWriter writer(cc);
  for(Alien::Integer i = 0; i < lsize; ++i)
    writer[i] = offset + i + shift;
};

TEST(TestCompositeMatrixToMCG, CompositeMCGTest) {
#ifdef ALIEN_USE_MCGSOLVER
  Alien::Integer size = 5;
  Alien::Integer nbSubMatrices = 2;
  Alien::MatrixDistribution dist(size, size, Environment::parallelMng());
  Alien::VectorDistribution vecDist(size, Environment::parallelMng());
  //Alien::CompositeMatrixData v(size*nbSubMatrices, nbSubMatrices); // matrice 10x10
  //Alien::CompositeVectorData vec(size*nbSubMatrices, nbSubMatrices); // vecteur 10x1
  //Alien::CompositeVectorData vecSol(size*nbSubMatrices, nbSubMatrices); // vecteur 10x1
  Alien::CompositeMatrix v ;
  v.resize(nbSubMatrices) ;
  Alien::CompositeVector vec ;
  vec.resize(nbSubMatrices) ;
  Alien::CompositeVector vecSol ;
  vecSol.resize(nbSubMatrices) ;
  for(Alien::Integer i=0; i<nbSubMatrices; ++i)
  {
    Alien::CompositeElement(vec,i) = Alien::Vector(vecDist);
    auto& vi = vec[i];
    ::fill(vi, 1.);
    Alien::CompositeElement(vecSol,i) = Alien::Vector(vecDist);
    auto& vSol = vecSol[i];
    ::fill(vSol, 0.);
    for(Alien::Integer j=0;j<nbSubMatrices; ++j)
    {
      Alien::CompositeElement(v,i,j) = Alien::Matrix(dist);
      auto& cij = v(i,j);
      auto& M = Alien::cast<Alien::Matrix>(cij);
      auto tag = Alien::DirectMatrixOptions::eResetValues;
      Alien::DirectMatrixBuilder builder(M, tag);
      builder.reserve(size);
      builder.allocate();
      for(Alien::Integer i = 0;i<size;++i)
      {
        builder(i,i) = 2;
        if(i>0)
          builder(i,i-1) = -0.1;
        if(i<size-1)
          builder(i,i+1) = -0.1;
      }
    }
  }

  ASSERT_TRUE(vec.hasUserFeature("composite"));
  ASSERT_EQ (nbSubMatrices, vec.size());
  ASSERT_EQ(nbSubMatrices*size, vec.space().size());
  ASSERT_TRUE(vecSol.hasUserFeature("composite"));
  ASSERT_EQ (nbSubMatrices, vecSol.size());
  ASSERT_EQ(nbSubMatrices*size, vecSol.space().size());
  ASSERT_TRUE(v.hasUserFeature("composite"));
  ASSERT_EQ (nbSubMatrices, v.size());
  ASSERT_EQ(nbSubMatrices*size, v.rowSpace().size());
  ASSERT_EQ(nbSubMatrices*size, v.colSpace().size());
  //
  using namespace GPUSolverOptionsNames;
  std::shared_ptr<IOptionsGPUSolver> options(new StrongOptionsGPUSolver{
	  _output = 1
	  , _maxIterationNum = 1000
	  , _stopCriteriaValue = 1e-6
	  , _kernel = GPUOptionTypes::eKernelType::MCKernel
	  , _solver = GPUOptionTypes::eSolver::BiCGStab
	  , _preconditioner = GPUOptionTypes::ePreconditioner::ILU0PC
  });
  auto* solver = new Alien::GPULinearSolver(Environment::parallelMng(), options);
  //
  solver->init();
  solver->solve(v,vec,vecSol);
  const Alien::SolverStatus & status = solver->getStatus() ;
  if(status.succeeded)
  {
    std::cout<<"Solver succeed   "<<status.succeeded<<std::endl ;
    std::cout<<"             residual "<<status.residual<<std::endl ;
    std::cout<<"             nb iters  "<<status.iteration_count<<std::endl ;
    std::cout<<"             error      "<<status.error<<std::endl ;
  }
  delete solver;
#endif
}
