
#include "alien/kernels/mcg/linear_solver/arcane/MCGLinearSolver.h"
#include "ALIEN/axl/MCGSolver_StrongOptions.h"

/**
 * Interface du service de résolution de système linéaire
 */

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
MCGLinearSolver::MCGLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneMCGSolverObject(sbi)
, Alien::MCGInternalLinearSolver(sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
  Alien::setTraceMng(sbi.subDomain()->traceMng());
}
#endif

MCGLinearSolver::MCGLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsMCGSolver> _options)
: ArcaneMCGSolverObject(_options)
, Alien::MCGInternalLinearSolver(parallel_mng, options())
{}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MCGSOLVER(MCGSolver, MCGLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_MCGSOLVER();
