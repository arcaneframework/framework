

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#include "HARTS/HARTS.h"
#include "HARTSSolver/HTS.h"
#include <alien/kernels/hts/linear_solver/arcane/HTSLinearSolver.h>
#include <ALIEN/axl/HTSSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
HTSLinearSolver::HTSLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneHTSSolverObject(sbi)
, Alien::HTSInternalLinearSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, LinearSolver<BackEnd::tag::htssolver>(sbi.subDomain()->parallelMng(), options())
{
}
#endif

HTSLinearSolver::HTSLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsHTSSolver> _options)
: ArcaneHTSSolverObject(_options)
, Alien::HTSInternalLinearSolver(parallel_mng, options())
//, LinearSolver<BackEnd::tag::htssolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HTSSOLVER(HTSSolver, HTSLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_HTSSOLVER();
