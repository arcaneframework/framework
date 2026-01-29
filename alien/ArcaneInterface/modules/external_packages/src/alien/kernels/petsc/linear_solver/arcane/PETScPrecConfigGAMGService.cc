// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/linear_solver/arcane/PETScPrecConfigGAMGService.h>
#include <ALIEN/axl/PETScPrecConfigGAMG_StrongOptions.h>

#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef ALIEN_USE_ARCANE
PETScPrecConfigGAMGService::PETScPrecConfigGAMGService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigGAMGObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif

PETScPrecConfigGAMGService::PETScPrecConfigGAMGService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScPrecConfigGAMG> options)
: ArcanePETScPrecConfigGAMGObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

bool
PETScPrecConfigGAMGService::needPrematureKSPSetUp() const
{
  return false;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
PETScPrecConfigGAMGService::configure(PC& pc,
                                          [[maybe_unused]] const ISpace& space,
                                          [[maybe_unused]] const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc GAMG preconditioner"; });
  checkError("Set preconditioner", PCSetType(pc, PCHMG));
  checkError("Set preconditioner", PCHMGSetInnerPCType(pc, PCGAMG));
  checkError("Set preconditioner", PCHMGSetReuseInterpolation(pc, PETSC_TRUE));
  checkError("Set preconditioner", PCHMGSetUseSubspaceCoarsening(pc, PETSC_TRUE));
  checkError("Set preconditioner", PCHMGUseMatMAIJ(pc, PETSC_FALSE));
  checkError("Set preconditioner", PCHMGSetCoarseningComponent(pc, 0));


  {
    checkError("Set GAMG stype",PetscOptionsSetValue(NULL, "-pc_gamg_type",options()->gamgType().localstr())) ;
  }

  /*
  if(options()->gamgMaxLevels()>0)
  {
    Arcane::String max_level = Arcane::String::format("{0}",options()->gamgMaxLevels()) ;
    checkError("Set GAMG max level", PetscOptionsSetValue(NULL, "-pc_mg_levels",max_level.localstr())) ;
  }*/

  if(options()->gamgThreshold()>0)
  {
    Arcane::String factor = Arcane::String::format("{0}",options()->gamgThreshold()) ;
    checkError("Set GAMG threshold", PetscOptionsSetValue(NULL, "-pc_gamg_threshold",factor.localstr())) ;
  }

  if(options()->gamgAggNsmooths()>0)
  {
    Arcane::String nsmooths = Arcane::String::format("{0}",options()->gamgAggNsmooths()) ;
    checkError("Set GAMG agg nsmooths", PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths",nsmooths.localstr())) ;
  }

  if(options()->gamgAggressiveCoarsening()>0)
  {
    Arcane::String nlevel = Arcane::String::format("{0}",options()->gamgAggressiveCoarsening()) ;
    checkError("Set GAMG agg nsmooths", PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening",nlevel.localstr())) ;
  }

  if(options()->gamgAggressiveSquareGraph()>=0)
  {
    Arcane::String square_graph = Arcane::String::format("{0}",options()->gamgAggressiveSquareGraph()) ;
    checkError("Set GAMG agg square graph", PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_square_graph",square_graph.localstr())) ;
  }

}
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGGAMG(GAMG, PETScPrecConfigGAMGService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGGAMG();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
