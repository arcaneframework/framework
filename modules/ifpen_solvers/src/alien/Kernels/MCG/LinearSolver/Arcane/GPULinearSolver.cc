
#include <alien/Kernels/MCG/LinearSolver/Arcane/GPULinearSolver.h>
#include <ALIEN/axl/GPUSolver_StrongOptions.h>

/**
 * Interface du service de résolution de système linéaire
 */

//class GPUSolver ;

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
GPULinearSolver::GPULinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneGPUSolverObject(sbi)
, Alien::MCGInternalLinearSolver(sbi.subDomain()->parallelMng(), options())
{}
#endif

GPULinearSolver::GPULinearSolver(IParallelMng* parallel_mng,  std::shared_ptr<IOptionsGPUSolver>  _options)
: ArcaneGPUSolverObject(_options)
, Alien::MCGInternalLinearSolver(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_GPUSOLVER(GPUSolver,GPULinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_GPUSOLVER();
