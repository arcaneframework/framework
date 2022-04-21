

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"

#include "alien/kernels/trilinos/TrilinosPrecomp.h"
#include "alien/AlienTrilinosPrecomp.h"

#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>

#include <alien/kernels/trilinos/linear_solver/TrilinosOptionTypes.h>
#include <ALIEN/axl/TrilinosSolver_StrongOptions.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosInternalSolver.h>
#include <alien/kernels/trilinos/linear_solver/TrilinosInternalLinearSolver.h>
#include <alien/kernels/trilinos/linear_solver/arcane/TrilinosLinearSolver.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
template <typename TagT>
TrilinosLinearSolver<TagT>::TrilinosLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneTrilinosSolverObject(sbi)
, LinearSolver<TagT>(sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
}
#endif
template <typename TagT>
TrilinosLinearSolver<TagT>::TrilinosLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsTrilinosSolver> _options)
: ArcaneTrilinosSolverObject(_options)
, LinearSolver<TagT>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

#ifdef KOKKOS_ENABLE_SERIAL
template class TrilinosLinearSolver<BackEnd::tag::tpetraserial>;
typedef TrilinosLinearSolver<BackEnd::tag::tpetraserial> SolverSerial;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolver, SolverSerial);
#endif

  
#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosLinearSolver<BackEnd::tag::tpetraomp>;
typedef TrilinosLinearSolver<BackEnd::tag::tpetraomp> SolverOMP;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolverOMP, SolverOMP);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosLinearSolver<BackEnd::tag::tpetrapth>;
typedef TrilinosLinearSolver<BackEnd::tag::tpetrapth> SolverPTH;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolverPTH, SolverPTH);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class TrilinosLinearSolver<BackEnd::tag::tpetracuda>;
typedef TrilinosLinearSolver<BackEnd::tag::tpetracuda> SolverCUDA;
ARCANE_REGISTER_SERVICE_TRILINOSSOLVER(TrilinosSolverCUDA, SolverCUDA);
#endif

} // namespace Alien

REGISTER_STRONG_OPTIONS_TRILINOSSOLVER();
