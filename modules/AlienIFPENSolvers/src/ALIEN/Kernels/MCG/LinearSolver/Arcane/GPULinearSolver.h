#ifndef GPUSOLVERIMPL_H
#define GPUSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <ALIEN/Kernels/MCG/MCGPrecomp.h>
#include <alien/utils/Precomp.h>
#include <ALIEN/Alien-IFPENSolversPrecomp.h>
#include <ALIEN/Kernels/MCG/MCGBackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <ALIEN/Kernels/MCG/LinearSolver/MCGInternalLinearSolver.h>
#include <ALIEN/Kernels/MCG/LinearSolver/GPUOptionTypes.h>
#include <ALIEN/axl/GPUSolver_axl.h>


/**
 * Interface du service de résolution de système linéaire
 */

//class GPUSolver ;

namespace Alien {

class ALIEN_IFPENSOLVERS_EXPORT GPULinearSolver : public ArcaneGPUSolverObject,
                                                  public Alien::MCGInternalLinearSolver
#ifdef ARCGEOSIM_COMP
,
                                                  public IInfoModel
#endif
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  GPULinearSolver(const Arcane::ServiceBuildInfo & sbi);
#endif

  GPULinearSolver(IParallelMng* parallel_mng, std::shared_ptr<IOptionsGPUSolver> _options);

  /** Destructeur de la classe */
  virtual ~GPULinearSolver(){};


};

} // namespace Alien
#endif /* GPUSOLVERIMPL_H */
