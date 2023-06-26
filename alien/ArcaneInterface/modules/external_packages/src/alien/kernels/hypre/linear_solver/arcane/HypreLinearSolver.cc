#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>

namespace Alien {
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ALIEN_USE_ARCANE
HypreLinearSolver::HypreLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneHypreSolverObject(sbi)
, LinearSolver<BackEnd::tag::hypre>(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
  ;
}
#endif

/*---------------------------------------------------------------------------*/

HypreLinearSolver::HypreLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsHypreSolver> _options)
: ArcaneHypreSolverObject(_options)
, LinearSolver<BackEnd::tag::hypre>(parallel_mng, options())
{
  ;
}

/*---------------------------------------------------------------------------*/

HypreLinearSolver::~HypreLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HYPRESOLVER(HypreSolver, HypreLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_HYPRESOLVER();
