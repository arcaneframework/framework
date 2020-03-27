

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <ALIEN/Kernels/Trilinos/TrilinosPrecomp.h>
#include <ALIEN/Alien-TrilinosPrecomp.h>

#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosInternal.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosMatrix.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>

#include <ALIEN/Kernels/Trilinos/EigenSolver/TrilinosEigenOptionTypes.h>
#include <ALIEN/axl/TrilinosEigenSolver_StrongOptions.h>
#include <ALIEN/Kernels/Trilinos/EigenSolver/TrilinosInternalEigenSolver.h>
#include <ALIEN/Kernels/Trilinos/EigenSolver/Arcane/TrilinosEigenSolver.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
TrilinosEigenSolver::TrilinosEigenSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneTrilinosEigenSolverObject(sbi)
, Alien::TrilinosInternalEigenSolver(sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(sbi.subDomain()->parallelMng(), options())
{}
#endif

    TrilinosEigenSolver::TrilinosEigenSolver(Arccore::MessagePassing::IMessagePassingMng *parallel_mng,
                                             std::shared_ptr<IOptionsTrilinosEigenSolver> _options)
            : ArcaneTrilinosEigenSolverObject(_options), Alien::TrilinosInternalEigenSolver(parallel_mng, options())
//, EigenSolver<BackEnd::tag::Trilinossolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_TRILINOSEIGENSOLVER(TrilinosEigenSolver,TrilinosEigenSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_TRILINOSEIGENSOLVER();
