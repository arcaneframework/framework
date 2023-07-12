#ifndef MTLSOLVERIMPL_H
#define MTLSOLVERIMPL_H

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/mtl/MTLPrecomp.h>
#include <alien/kernels/mtl/linear_solver/MTLInternalLinearSolver.h>

#include <alien/kernels/mtl/MTLBackEnd.h>
#include <alien/core/backend/LinearSolver.h>

#include <alien/kernels/mtl/linear_solver/MTLOptionTypes.h>

#include <ALIEN/axl/MTLLinearSolver_axl.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

class ALIEN_EXTERNAL_PACKAGES_EXPORT MTLLinearSolverService
    : public ArcaneMTLLinearSolverObject,
      public LinearSolver<BackEnd::tag::mtl>
{
 private:
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  MTLLinearSolverService(const Arcane::ServiceBuildInfo& sbi);
#endif
  MTLLinearSolverService(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsMTLLinearSolver> options);
  /** Destructeur de la classe */
  virtual ~MTLLinearSolverService();
  friend class MTLLinearSystem;

 public:
  //! Initialisation
  // void init() ;

  //! Finalize
};

} // namespace Alien

#endif // MTLSOLVERIMPL_H
