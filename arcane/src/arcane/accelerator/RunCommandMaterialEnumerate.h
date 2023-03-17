// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandMaterialEnumerate.h                               (C) 2000-2022 */
/*                                                                           */
/* Helpers et macros pour exécuter une boucle sur une liste d'envcell        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Concurrency.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MeshMaterialVariableIndexer.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/accelerator/RunQueueInternal.h"
#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe helper pour l'accès au MatVarIndex et au CellLocalId à travers les 
 *        RUNCOMMAND_ENUMERATE(EnvCell...
 */
class EnvCellAccessor
{
 public:
  ///! Struct interne simple pour éviter l'usage d'un std::tuple pour l'opérateur()
  struct EnvCellAccessorInternalData
  {
    MatVarIndex m_mvi;
    CellLocalId m_cid;   
  };

 public:
  inline ARCCORE_HOST_DEVICE explicit EnvCellAccessor(MatVarIndex mvi, CellLocalId cid)
  : m_internal_data{mvi, cid}
  {
  }

  /*!
  * \brief Cet opérateur permet de renvoyer le couple [MatVarIndex, LocalCellId].
  *
  * L'utilisation classique est :
  *         cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, envcellsv) {
  *         auto [mvi, cid] = evi();
  * où evi est de type EnvCellAccessor
  */
  inline ARCCORE_HOST_DEVICE auto operator()()
  {
    return EnvCellAccessorInternalData{m_internal_data.m_mvi, m_internal_data.m_cid};
  }
  
  ///! Accesseur sur la partie MatVarIndex
  ARCCORE_HOST_DEVICE MatVarIndex varIndex() { return m_internal_data.m_mvi; };
  
  ///! Accesseur sur la partie cell local id
  ARCCORE_HOST_DEVICE CellLocalId globalCellId() { return m_internal_data.m_cid; }
 
 private:
  EnvCellAccessorInternalData m_internal_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

/*
 * Surcharge de la fonction de lancement de kernel pour GPU pour les MatVarIndex et CellLocalId
 */ 
template<typename Lambda> __global__
void doIndirectGPULambda(SmallSpan<const MatVarIndex> mvis, SmallSpan<const Int32> cids, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<mvis.size()){
    EnvCellAccessor lec(mvis[i], static_cast<CellLocalId>(cids[i]));
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(lec);
  }
}

template<typename Lambda> __global__
void doDirectGPULambda(MatVarIndex mvi, Int32 cid, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (!mvi.null()){
    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(EnvCellAccessor(mvi, static_cast<CellLocalId>(cid)));
  }
}

#endif // ARCANE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Surcharge de la fonction de lancement de kernel en MT pour les EnvCellVectorView
 */ 
template<typename Lambda>
void doIndirectThreadLambda(SmallSpan<const MatVarIndex>& sub_mvis, SmallSpan<const Int32> sub_cids, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  // Les tailles de sub_mvis et sub_cids ont été testées en amont déjà
  for (int i(0); i<sub_mvis.size(); ++i)
    body(EnvCellAccessor(sub_mvis[i], static_cast<CellLocalId>(sub_cids[i])));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 *        Uniquement pour les EnvCellVectorView
 */
template<typename Lambda> void
_applyEnvCells(RunCommand& command,const EnvCellVectorView& items,const Lambda& func)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = static_cast<Int32>(items.nbItem());
  if (vsize==0)
    return;

  SmallSpan<const MatVarIndex> mvis(items.matvarIndexes());
  SmallSpan<const Int32> cids(items._internalLocalIds());
  ARCANE_ASSERT(mvis.size() == cids.size(), ("MatVarIndex and CellLocalId arrays have different size"));

  RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.computeLoopRunInfo(vsize);
  launch_info.beginExecute();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info,ARCANE_KERNEL_CUDA_FUNC(doIndirectGPULambda)<Lambda>,func,mvis,cids);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info,ARCANE_KERNEL_HIP_FUNC(doIndirectGPULambda)<Lambda>,func,mvis,cids);
    break;
  case eExecutionPolicy::Sequential:
      for (int i(0); i<mvis.size(); ++i)
        func(EnvCellAccessor(mvis[i], static_cast<CellLocalId>(cids[i])));
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelForVa(
                        launch_info.loopRunInfo(),
                        [&](SmallSpan<const MatVarIndex> sub_mvis, SmallSpan<const Int32> sub_cids)
                        {
                          doIndirectThreadLambda(sub_mvis, sub_cids,func);
                        }, mvis, cids);
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'",exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

///! Spécialization du run pour les EnvCellVectorView
template<typename Lambda> void
run(RunCommand& command,const EnvCellVectorView& items,const Lambda& func)
{
  impl::_applyEnvCells(command,items,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Equivalent de la classe ItemRunCommand pour les EnvCell
 */
class EnvCellRunCommand
{
 public:
  explicit EnvCellRunCommand(RunCommand& command,const EnvCellVectorView& items)
  : m_command(command), m_items(items)
  {
  }

  RunCommand& m_command;
  EnvCellVectorView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_EXPORT EnvCellRunCommand
operator<<(RunCommand& command,const EnvCellVectorView& items);

extern "C++" ARCANE_ACCELERATOR_EXPORT EnvCellRunCommand
operator<<(RunCommand& command,IMeshEnvironment* env);

template<typename Lambda>
void operator<<(EnvCellRunCommand&& nr,const Lambda& f)
{
  run(nr.m_command,nr.m_items,f);
}

template<typename Lambda>
void operator<<(EnvCellRunCommand& nr,const Lambda& f)
{
  run(nr.m_command,nr.m_items,f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// FIXME: Hack assez moche pour tester au plus vite, il faudra voir comment faire ça propre
namespace Arcane {
  namespace Materials {
    using EnvCellLocalId = Arcane::Accelerator::EnvCellAccessor;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
