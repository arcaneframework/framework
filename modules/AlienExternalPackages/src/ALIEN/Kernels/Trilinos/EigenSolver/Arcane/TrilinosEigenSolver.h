#ifndef TRILINOSEIGENSOLVERIMPL_H
#define TRILINOSEIGENSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <ALIEN/Kernels/Trilinos/TrilinosPrecomp.h>
#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Alien-ExternalPackagesPrecomp.h>
#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Core/Backend/EigenSolver.h>
#include <ALIEN/Kernels/Trilinos/EigenSolver/TrilinosInternalEigenSolver.h>
#include <ALIEN/Kernels/Trilinos/EigenSolver/TrilinosEigenOptionTypes.h>
#include <ALIEN/axl/TrilinosEigenSolver_axl.h>


/**
 * Interface du service de résolution de probleme aux valuers propres
 */

//class TrilinosSolver ;

namespace Alien {

class ALIEN_EXTERNALPACKAGES_EXPORT TrilinosEigenSolver
: public ArcaneTrilinosEigenSolverObject,
  public TrilinosInternalEigenSolver
//, public LinearSolver<BackEnd::tag::Trilinossolver>
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  TrilinosEigenSolver(const Arcane::ServiceBuildInfo & sbi);
#endif

  TrilinosEigenSolver(IParallelMng* parallel_mng, std::shared_ptr<IOptionsTrilinosEigenSolver> _options);

  /** Destructeur de la classe */
  virtual ~TrilinosEigenSolver(){};


};

} // namespace Alien
#endif /* TRILINOSSOLVERIMPL_H */
