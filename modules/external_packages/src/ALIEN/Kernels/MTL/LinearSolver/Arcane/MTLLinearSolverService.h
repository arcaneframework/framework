#ifndef MTLSOLVERIMPL_H
#define MTLSOLVERIMPL_H

#include <ALIEN/AlienExternalPackagesPrecomp.h>
#include <ALIEN/Kernels/MTL/MTLPrecomp.h>
#include <ALIEN/Kernels/MTL/LinearSolver/MTLInternalLinearSolver.h>

#include <ALIEN/Kernels/MTL/MTLBackEnd.h>
#include <alien/core/backend/LinearSolver.h>

#include <ALIEN/Kernels/MTL/LinearSolver/MTLOptionTypes.h>

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
  MTLLinearSolverService(const Arcane::ServiceBuildInfo & sbi);
#endif
  MTLLinearSolverService(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsMTLLinearSolver> options);
  /** Destructeur de la classe */
  virtual ~MTLLinearSolverService();
  friend class MTLLinearSystem ; 
public:
  //! Initialisation
  //void init() ;

  //! Finalize
};

} // namespace Alien

#endif //MTLSOLVERIMPL_H
