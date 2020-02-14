

/**
 * Interface du service de résolution de système linéaire
 */
#include "ALIEN/ALIENConfig.h"
#include <ALIEN/Kernels/Trilinos/EigenSolver/Arcane/TrilinosEigenSolver.h>
#include <ALIEN/axl/TrilinosEigenSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
TrilinosEigenSolver::TrilinosEigenSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneTrilinosEigenSolverObject(sbi)
, Alien::TrilinosInternalEigenSolver(sbi.subDomain()->parallelMng(), options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(sbi.subDomain()->parallelMng(), options())
{}
#endif

TrilinosEigenSolver::TrilinosEigenSolver(IParallelMng* parallel_mng,  std::shared_ptr<IOptionsTrilinosEigenSolver>  _options)
: ArcaneTrilinosEigenSolverObject(_options)
, Alien::TrilinosInternalEigenSolver(parallel_mng, options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TRILINOSEIGENSOLVER(TrilinosEigenSolver,TrilinosEigenSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_TRILINOSEIGENSOLVER();
