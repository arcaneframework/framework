#ifndef HTSEIGENSOLVERIMPL_H
#define HTSEIGENSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <ALIEN/Kernels/HTS/HTSPrecomp.h>
#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Alien-IFPENSolversPrecomp.h>
#include <ALIEN/Kernels/HTS/HTSBackEnd.h>
#include <ALIEN/Core/Backend/EigenSolver.h>
#include <ALIEN/Kernels/HTS/EigenSolver/HTSInternalEigenSolver.h>
#include <ALIEN/Kernels/HTS/EigenSolver/HTSEigenOptionTypes.h>
#include <ALIEN/axl/HTSEigenSolver_axl.h>


/**
 * Interface du service de résolution de probleme aux valuers propres
 */

//class HTSSolver ;

namespace Alien {

class ALIEN_IFPENSOLVERS_EXPORT HTSEigenSolver : public ArcaneHTSEigenSolverObject,
                                                 public HTSInternalEigenSolver
//, public LinearSolver<BackEnd::tag::htssolver>
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  HTSEigenSolver(const Arcane::ServiceBuildInfo & sbi);
#endif

  HTSEigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsHTSEigenSolver> _options);

  /** Destructeur de la classe */
  virtual ~HTSEigenSolver(){};


};

} // namespace Alien
#endif /* HTSSOLVERIMPL_H */
