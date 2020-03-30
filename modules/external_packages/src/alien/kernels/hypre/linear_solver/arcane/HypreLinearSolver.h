#ifndef HYPRESOLVERIMPL_H
#define HYPRESOLVERIMPL_H

#include <alien/utils/Precomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/core/backend/LinearSolver.h>

#include <alien/kernels/hypre/linear_solver/HypreOptionTypes.h>
#include <ALIEN/axl/HypreSolver_axl.h>

namespace Alien {

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreLinearSolver
    : public ArcaneHypreSolverObject,
      public LinearSolver<BackEnd::tag::hypre>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  HypreLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  HypreLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsHypreSolver> _options);

  /** Destructeur de la classe */
  virtual ~HypreLinearSolver();

 public:
};

} // namespace Alien

#endif /* HYPRESOLVERIMPL_H */
