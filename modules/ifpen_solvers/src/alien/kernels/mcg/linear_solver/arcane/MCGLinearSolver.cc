
#include <ALIEN/Kernels/MCG/LinearSolver/Arcane/MCGLinearSolver.h>
#include <ALIEN/axl/MCGSolver_StrongOptions.h>

/**
 * Interface du service de résolution de système linéaire
 */

BEGIN_NAMESPACE(Alien)

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
MCGLinearSolver::MCGLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneGPUSolverObject(sbi)
, Alien::MCGInternalLinearSolver(sbi.subDomain()->parallelMng(), options())
{
}
#endif

MCGLinearSolver::MCGLinearSolver(
    IParallelMng* parallel_mng, std::shared_ptr<IOptionsMCGSolver> _options)
: ArcaneMCGSolverObject(_options)
, Alien::MCGInternalLinearSolver(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MCGSOLVER(MCGSolver, MCGLinearSolver);

END_NAMESPACE

REGISTER_STRONG_OPTIONS_MCGSOLVER();
