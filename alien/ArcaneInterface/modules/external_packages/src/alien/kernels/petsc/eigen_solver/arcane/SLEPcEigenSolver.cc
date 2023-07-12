

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#include <alien/kernels/petsc/eigen_solver/arcane/SLEPcEigenSolver.h>
#include <ALIEN/axl/SLEPcEigenSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
SLEPcEigenSolver::SLEPcEigenSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneSLEPcEigenSolverObject(sbi)
, Alien::SLEPcInternalEigenSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, EigenSolver<BackEnd::tag::SLEPcsolver>(sbi.subDomain()->parallelMng(), options())
{
}
#endif

SLEPcEigenSolver::SLEPcEigenSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsSLEPcEigenSolver> _options)
: ArcaneSLEPcEigenSolverObject(_options)
, Alien::SLEPcInternalEigenSolver(parallel_mng, options())
//, EigenSolver<BackEnd::tag::SLEPcsolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SLEPCEIGENSOLVER(SLEPcEigenSolver, SLEPcEigenSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_SLEPCEIGENSOLVER();
