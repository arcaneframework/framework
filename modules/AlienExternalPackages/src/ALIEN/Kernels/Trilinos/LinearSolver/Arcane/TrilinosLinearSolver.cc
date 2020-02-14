

/**
 * Interface du service de résolution de système linéaire
 */
#include "ALIEN/ALIENConfig.h"

#include "ALIEN/Kernels/Trilinos/TrilinosPrecomp.h"

#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosInternal.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosMatrix.h>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>

#include <ALIEN/Kernels/Trilinos/LinearSolver/TrilinosOptionTypes.h>
#include <ALIEN/axl/TrilinosSolver_StrongOptions.h>
#include <ALIEN/Kernels/Trilinos/LinearSolver/TrilinosInternalSolver.h>
#include <ALIEN/Kernels/Trilinos/LinearSolver/TrilinosInternalLinearSolver.h>
#include <ALIEN/Kernels/Trilinos/LinearSolver/Arcane/TrilinosLinearSolver.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
template <typename TagT>
TrilinosLinearSolver<TagT>::TrilinosLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneTrilinosSolverObject(sbi)
, LinearSolver<TagT>(sbi.subDomain()->parallelMng(), options())
{}
#endif
template<typename TagT>
TrilinosLinearSolver<TagT>::TrilinosLinearSolver(IParallelMng* parallel_mng,  std::shared_ptr<IOptionsTrilinosSolver>  _options)
: ArcaneTrilinosSolverObject(_options)
, LinearSolver<TagT>(parallel_mng, options())
{
}


/*---------------------------------------------------------------------------*/

template class TrilinosLinearSolver<BackEnd::tag::tpetraserial> ;
typedef TrilinosLinearSolver<BackEnd::tag::tpetraserial> SolverSerial ;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolver,SolverSerial);

#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosLinearSolver<BackEnd::tag::tpetraomp> ;
typedef TrilinosLinearSolver<BackEnd::tag::tpetraomp> SolverOMP ;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolverOMP,SolverOMP);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosLinearSolver<BackEnd::tag::tpetrapth> ;
typedef TrilinosLinearSolver<BackEnd::tag::tpetrapth> SolverPTH ;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolverPTH,SolverPTH);
#endif
} // namespace Alien

REGISTER_STRONG_OPTIONS_TRILINOSSOLVER();
