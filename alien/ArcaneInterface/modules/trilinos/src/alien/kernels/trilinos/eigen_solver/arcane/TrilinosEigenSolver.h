#ifndef TRILINOSEIGENSOLVERIMPL_H
#define TRILINOSEIGENSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/kernels/trilinos/TrilinosPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienTrilinosPrecomp.h>
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/eigen_solver/TrilinosEigenOptionTypes.h>
#include <ALIEN/axl/TrilinosEigenSolver_axl.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/kernels/trilinos/eigen_solver/TrilinosInternalEigenSolver.h>

/**
 * Interface du service de résolution de probleme aux valuers propres
 */

// class TrilinosSolver ;

namespace Alien {

class ALIEN_TRILINOS_EXPORT TrilinosEigenSolver : public ArcaneTrilinosEigenSolverObject,
                                                  public TrilinosInternalEigenSolver
//, public LinearSolver<BackEnd::tag::Trilinossolver>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  TrilinosEigenSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  TrilinosEigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsTrilinosEigenSolver> _options);

  /** Destructeur de la classe */
  virtual ~TrilinosEigenSolver(){};
};

} // namespace Alien
#endif /* TRILINOSSOLVERIMPL_H */
