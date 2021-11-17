

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <alien/kernels/trilinos/TrilinosPrecomp.h>
#include <alien/AlienTrilinosPrecomp.h>

#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>

#include <alien/kernels/trilinos/eigen_solver/TrilinosEigenOptionTypes.h>
#include <ALIEN/axl/TrilinosEigenSolver_StrongOptions.h>
#include <alien/kernels/trilinos/eigen_solver/TrilinosInternalEigenSolver.h>
#include <alien/kernels/trilinos/eigen_solver/arcane/TrilinosEigenSolver.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
TrilinosEigenSolver::TrilinosEigenSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneTrilinosEigenSolverObject(sbi)
, Alien::TrilinosInternalEigenSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(sbi.subDomain()->parallelMng(), options())
{
}
#endif

TrilinosEigenSolver::TrilinosEigenSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsTrilinosEigenSolver> _options)
: ArcaneTrilinosEigenSolverObject(_options)
, Alien::TrilinosInternalEigenSolver(parallel_mng, options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TRILINOSEIGENSOLVER(TrilinosEigenSolver, TrilinosEigenSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_TRILINOSEIGENSOLVER();
