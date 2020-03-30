#ifndef HTSEIGENSOLVERIMPL_H
#define HTSEIGENSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/kernels/hts/HTSPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienIFPENSolversPrecomp.h>
#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/kernels/hts/eigen_solver/HTSInternalEigenSolver.h>
#include <alien/kernels/hts/eigen_solver/HTSEigenOptionTypes.h>
#include <ALIEN/axl/HTSEigenSolver_axl.h>

/**
 * Interface du service de résolution de probleme aux valuers propres
 */

// class HTSSolver ;

namespace Alien {

class ALIEN_IFPEN_SOLVERS_EXPORT HTSEigenSolver : public ArcaneHTSEigenSolverObject,
                                                  public HTSInternalEigenSolver
//, public LinearSolver<BackEnd::tag::htssolver>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  HTSEigenSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  HTSEigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsHTSEigenSolver> _options);

  /** Destructeur de la classe */
  virtual ~HTSEigenSolver(){};
};

} // namespace Alien
#endif /* HTSSOLVERIMPL_H */
