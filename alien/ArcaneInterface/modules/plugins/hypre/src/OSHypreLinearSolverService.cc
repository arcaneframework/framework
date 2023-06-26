// backend opensource
#include <alien/hypre/backend.h>
#include <alien/hypre/options.h>
#include <ALIEN/axl/OSHypreSolver_axl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OSHypreLinearSolverService : public ArcaneOSHypreSolverObject
{
 public:
  OSHypreLinearSolverService(const Arcane::ServiceBuildInfo& sbi)
  : ArcaneOSHypreSolverObject(sbi)
  {
  }

  virtual ~OSHypreLinearSolverService() {}

  void init()
  {
    auto solveroptions = Alien::Hypre::Options()
                             .numIterationsMax(options()->numIterationsMax())
                             .stopCriteriaValue(options()->stopCriteriaValue())
                             .preconditioner(options()->preconditioner())
                             .solver(options()->solver());

    m_solver.reset(new Alien::Hypre::LinearSolver(solveroptions));
  }

  Arccore::String getBackEndName() const { return m_solver->getBackEndName(); }

  void end() { m_solver->end(); }

  bool solve(const Alien::IMatrix& A, const Alien::IVector& b, Alien::IVector& x)
  {
    return m_solver->solve(A, b, x);
  }

  Alien::SolverStat const& getSolverStat() const { return m_solver->getSolverStat(); }

  std::shared_ptr<Alien::ILinearAlgebra> algebra() const { return m_solver->algebra(); }

  bool hasParallelSupport() const { return m_solver->hasParallelSupport(); }

  const Alien::SolverStatus& getStatus() const { return m_solver->getStatus(); }

  void setNullSpaceConstantOption(bool flag)
  {
    m_solver->setNullSpaceConstantOption(flag);
  }

 private:
  std::unique_ptr<Alien::Hypre::LinearSolver> m_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_OSHYPRESOLVER(OSHypreSolver, OSHypreLinearSolverService);
