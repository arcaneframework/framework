#ifndef HTSSOLVERIMPL_H
#define HTSSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <ALIEN/Kernels/HTS/HTSPrecomp.h>
#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Alien-IFPENSolversPrecomp.h>
#include <ALIEN/Kernels/HTS/HTSBackEnd.h>
#include <ALIEN/Core/Backend/LinearSolver.h>
#include <ALIEN/Kernels/HTS/LinearSolver/HTSInternalLinearSolver.h>
#include <ALIEN/Kernels/HTS/LinearSolver/HTSOptionTypes.h>
#include <ALIEN/axl/HTSSolver_axl.h>


/**
 * Interface du service de résolution de système linéaire
 */

//class HTSSolver ;

namespace Alien {

class ALIEN_IFPENSOLVERS_EXPORT HTSLinearSolver : public ArcaneHTSSolverObject,
                                                  public HTSInternalLinearSolver
//, public LinearSolver<BackEnd::tag::htssolver>
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  HTSLinearSolver(const Arcane::ServiceBuildInfo & sbi);
#endif

  HTSLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsHTSSolver> _options);

  /** Destructeur de la classe */
  virtual ~HTSLinearSolver(){};


};

} // namespace Alien
#endif /* HTSSOLVERIMPL_H */
