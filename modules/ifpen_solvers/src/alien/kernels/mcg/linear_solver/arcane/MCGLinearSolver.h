#ifndef MCGSOLVERIMPL_H
#define MCGSOLVERIMPL_H

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <ALIEN/Kernels/MCG/MCGPrecomp.h>
#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Alien-ExternalPackagesPrecomp.h>
#include <ALIEN/Kernels/MCG/MCGBackEnd.h>
#include <ALIEN/Core/Backend/LinearSolver.h>
#include <ALIEN/Kernels/MCG/LinearSolver/MCGInternalLinearSolver.h>
#include <ALIEN/Kernels/MCG/LinearSolver/MCGOptionTypes.h>
#include <ALIEN/axl/MCGSolver_axl.h>

/**
 * Interface du service de résolution de système linéaire
 */

BEGIN_NAMESPACE(Alien)

class ALIEN_EXTERNALPACKAGES_EXPORT MCGLinearSolver
    : public ArcaneMCGSolverObject,
      public Alien::MCGInternalLinearSolver
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

  MCGLinearSolver(
      IParallelMng* parallel_mng, std::shared_ptr<IOptionsMCGSolver> _options);

  /** Destructeur de la classe */
  virtual ~MCGLinearSolver(){};
};

END_NAMESPACE
#endif /* MCGSOLVERIMPL_H */
