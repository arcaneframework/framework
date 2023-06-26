#ifndef MCGSOLVERIMPL_H
#define MCGSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/utils/Precomp.h>
#include <alien/core/backend/LinearSolver.h>

#include "alien/AlienIFPENSolversPrecomp.h"
#include "alien/kernels/mcg/MCGPrecomp.h"
#include "alien/kernels/mcg/MCGBackEnd.h"
#include "alien/kernels/mcg/linear_solver/MCGInternalLinearSolver.h"
#include "alien/kernels/mcg/linear_solver/MCGOptionTypes.h"
#include "ALIEN/axl/MCGSolver_axl.h"

/**
 * Interface du service de résolution de système linéaire
 */

namespace Alien {

class ALIEN_IFPEN_SOLVERS_EXPORT MCGLinearSolver : public ArcaneMCGSolverObject,
                                                   public MCGInternalLinearSolver
#ifdef ARCGEOSIM_COMP
,
                                                   public IInfoModel
#endif
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  MCGLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  MCGLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsMCGSolver> _options);

  /** Destructeur de la classe */
  virtual ~MCGLinearSolver(){};
};
} // namespace Alien

#endif /* MCGSOLVERIMPL_H */
