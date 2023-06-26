#ifndef HTSSOLVERIMPL_H
#define HTSSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/kernels/hts/HTSPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienIFPENSolversPrecomp.h>
#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/kernels/hts/linear_solver/HTSInternalLinearSolver.h>
#include <alien/kernels/hts/linear_solver/HTSOptionTypes.h>
#include <ALIEN/axl/HTSSolver_axl.h>

/**
 * Interface du service de résolution de système linéaire
 */

// class HTSSolver ;

namespace Alien {

class ALIEN_IFPEN_SOLVERS_EXPORT HTSLinearSolver : public ArcaneHTSSolverObject,
                                                   public HTSInternalLinearSolver
//, public LinearSolver<BackEnd::tag::htssolver>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  HTSLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  HTSLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsHTSSolver> _options);

  /** Destructeur de la classe */
  virtual ~HTSLinearSolver(){};
};

} // namespace Alien
#endif /* HTSSOLVERIMPL_H */
