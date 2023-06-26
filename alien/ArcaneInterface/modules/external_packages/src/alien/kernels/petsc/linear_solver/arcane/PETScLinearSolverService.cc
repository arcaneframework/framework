/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/kernels/petsc/linear_solver/arcane/PETScLinearSolverService.h>
#include <ALIEN/axl/PETScLinearSolver_StrongOptions.h>

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef ALIEN_USE_ARCANE
PETScLinearSolverService::PETScLinearSolverService(const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScLinearSolverObject(sbi)
, LinearSolver<BackEnd::tag::petsc>(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
  ;
}
#endif
/*---------------------------------------------------------------------------*/

PETScLinearSolverService::PETScLinearSolverService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScLinearSolver> _options)
: ArcanePETScLinearSolverObject(_options)
, LinearSolver<BackEnd::tag::petsc>(parallel_mng, options())
{
  ;
}

/*---------------------------------------------------------------------------*/

PETScLinearSolverService::~PETScLinearSolverService()
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCLINEARSOLVER(PETScSolver, PETScLinearSolverService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCLINEARSOLVER();
