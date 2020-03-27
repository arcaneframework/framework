#ifndef TRILINOSSOLVERIMPL_H
#define TRILINOSSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/Kernels/Trilinos/TrilinosPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienTrilinosPrecomp.h>
#include <alien/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/Kernels/Trilinos/LinearSolver/TrilinosInternalLinearSolver.h>
#include <alien/Kernels/Trilinos/LinearSolver/TrilinosOptionTypes.h>
#include <ALIEN/axl/TrilinosSolver_axl.h>


/**
 * Interface du service de résolution de système linéaire
 */

//class TrilinosSolver ;

namespace Alien {

template <typename TagT>
class ALIEN_TRILINOS_EXPORT TrilinosLinearSolver
: public ArcaneTrilinosSolverObject,
  public TrilinosInternalLinearSolver<TagT>,
  public LinearSolver<TagT>
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  TrilinosLinearSolver(const Arcane::ServiceBuildInfo & sbi);
#endif

  TrilinosLinearSolver(Arccore::MessagePassing::IMessagePassingMng *parallel_mng,
                       std::shared_ptr<IOptionsTrilinosSolver> _options);

  /** Destructeur de la classe */
  virtual ~TrilinosLinearSolver(){};


};

} // namespace Alien
#endif /* TRILINOSSOLVERIMPL_H */
