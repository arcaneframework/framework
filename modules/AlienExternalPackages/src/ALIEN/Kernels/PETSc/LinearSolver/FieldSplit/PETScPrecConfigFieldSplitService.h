/* Author : gratienj
 * Preconditioner created by combining separate preconditioners for individual
 * fields or groups of fields. See the users manual section "Solving Block Matrices"
 * for more details in PETSc 3.3 documentation : 
 * http://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf
 */

#ifndef PETSCSOLVERCONFIGFIELDSPLITSERVICE_H
#define PETSCSOLVERCONFIGFIELDSPLITSERVICE_H

#include <ALIEN/Kernels/PETSc/PETScPrecomp.h>
#include <ALIEN/Alien-ExternalPackagesPrecomp.h>

#include <ALIEN/Kernels/PETSc/LinearSolver/IPETScPC.h>
#include <ALIEN/Kernels/PETSc/LinearSolver/IPETScKSP.h>
#include <ALIEN/Kernels/PETSc/LinearSolver/PETScConfig.h>
#include <ALIEN/Kernels/PETSc/LinearSolver/FieldSplit/IFieldSplitType.h>

#include <ALIEN/axl/PETScPrecConfigFieldSplit_axl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNALPACKAGES_EXPORT PETScPrecConfigFieldSplitService
: public ArcanePETScPrecConfigFieldSplitObject,
  public PETScConfig
{
 public:
  /** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScPrecConfigFieldSplitService(const Arcane::ServiceBuildInfo & sbi);
#endif

  PETScPrecConfigFieldSplitService(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScPrecConfigFieldSplit> options);
  /** Destructeur de la classe */
  virtual ~PETScPrecConfigFieldSplitService() {}
  
public:
  //! Initialisation
  void configure(PC & pc,
                 const ISpace& space,
                 const MatrixDistribution& distribution);

  //! Check need of KSPSetUp before calling this PC configure
   virtual bool needPrematureKSPSetUp() const { return true; }

 private:
  Arccore::Integer initializeFields(
      const ISpace& space, const MatrixDistribution& distribution);

 private:
  Arccore::String m_default_block_tag;
  Arccore::UniqueArray<Arccore::String> m_field_tags;
  Arccore::UniqueArray<IS> m_field_petsc_indices;
  Arccore::UniqueArray<PC> m_subpc;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //PETSCSOLVERCONFIGFIELDSPLITSERVICE_H
