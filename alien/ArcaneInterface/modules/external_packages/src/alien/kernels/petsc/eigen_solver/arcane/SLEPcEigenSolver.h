#ifndef SLEPCEIGENSOLVERIMPL_H
#define SLEPCEIGENSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/kernels/petsc/eigen_solver/SLEPcInternalEigenSolver.h>
#include <alien/kernels/petsc/eigen_solver/SLEPcEigenOptionTypes.h>
#include <ALIEN/axl/SLEPcEigenSolver_axl.h>

/**
 * Interface du service de résolution de probleme aux valuers propres
 */

// class SLEPcSolver ;

namespace Alien {

class ALIEN_EXTERNAL_PACKAGES_EXPORT SLEPcEigenSolver
    : public ArcaneSLEPcEigenSolverObject,
      public SLEPcInternalEigenSolver
//, public LinearSolver<BackEnd::tag::SLEPcsolver>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  SLEPcEigenSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  SLEPcEigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsSLEPcEigenSolver> _options);

  /** Destructeur de la classe */
  virtual ~SLEPcEigenSolver(){};
};

} // namespace Alien
#endif /* SLEPcSOLVERIMPL_H */
