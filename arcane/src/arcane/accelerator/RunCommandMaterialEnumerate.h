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

#include <arcane/core/materials/ComponentItemVectorView.h>
#include <arcane/core/materials/MaterialsCoreGlobal.h>
#include <arcane/core/materials/MatItem.h>
#include <arcane/accelerator/RunCommand.h>
#include <arcane/accelerator/RunCommandLaunchInfo.h>

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
// TODO: Si on garde cette classe il faudra la blinder, surtout sur le ctor component
// TODO: Faut(il revoir la classe pour faire qqch similaire à un Enumerator ?
class EnvCellAccessor {
 public:
  ARCCORE_HOST_DEVICE explicit EnvCellAccessor(EnvCell ec)
  : m_mvi(ec._varIndex()), m_cid(ec.globalCell().itemLocalId()) {}
  
  explicit EnvCellAccessor(ComponentItemInternal* cii)
  : m_mvi(cii->variableIndex()), m_cid(cii->globalItem()->localId()) {}

  ARCCORE_HOST_DEVICE explicit EnvCellAccessor(MatVarIndex mvi, CellLocalId cid)
  : m_mvi(MatVarIndex(mvi.arrayIndex(),mvi.valueIndex())), m_cid(cid) {}

  ARCCORE_HOST_DEVICE auto operator()() {
    return std::make_tuple(m_mvi, m_cid);
  }
  
  ARCCORE_HOST_DEVICE MatVarIndex varIndex() {return m_mvi;};
  
  ARCCORE_HOST_DEVICE CellLocalId globalCellId() {return m_cid;}
 
 private:
  MatVarIndex m_mvi;
  CellLocalId m_cid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

// Cet alias est nécessaire pour éviter les problèmes d'inférence de type
// dans les template avec des const * const ...
using ComponentItemInternalPtr = ComponentItemInternal*; 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

/*
 * Spécialization <MatVarIndex, CellLocalId> de la fonction de lancement de kernel pour GPU
 */ 
template<typename Lambda> __global__
void doIndirectGPULambda<Lambda>(SmallSpan<const MatVarIndex> mvis,SmallSpan<const Int32> cids,Lambda func)
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

#endif // ARCANE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Spécialization EnvCellVectorView de la fonction de lancement de kernel en MT
 */ 
template<typename Lambda>
void _doIndirectThreadLambda(const EnvCellVectorView& sub_items,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

// TODO: A valider avec GG si l'utilisation d'un for range est acceptable
  for (auto i : sub_items.itemsInternalView())
    body(EnvCellAccessor(i));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 *        Spécialization pour les EnvCellVectorView
 */
template<typename Lambda> void
_applyEnvCells(RunCommand& command,const EnvCellVectorView& items,const Lambda& func)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = static_cast<Int32>(items.nbItem());
  if (vsize==0)
    return;
  impl::RunCommandLaunchInfo launch_info(command);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
#if defined(ARCANE_COMPILING_CUDA)
    {
      launch_info.beginExecute();
      SmallSpan<const MatVarIndex> mvis(items.component()->variableIndexer()->matvarIndexes());
      SmallSpan<const Int32> cids(items.component()->variableIndexer()->localIds());
      // TODO: vérifier que l'arcane assert n'est pas tout le temps fait
      // ARCANE_ASSERT(mvis.size() == cids.size(), "MatVarIndex and CellLocalId arrays have different size");
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
      // TODO: le memadvise est mal placé ici, mais n'apporte pas plus dans l'indexer
      /*
      auto err1 = cudaMemAdvise(mvis.data(), mvis.sizeBytes(), cudaMemoryAdvise::cudaMemAdviseSetReadMostly, 0);
      auto err2 = cudaMemAdvise(cids.data(), cids.sizeBytes(), cudaMemoryAdvise::cudaMemAdviseSetReadMostly, 0);
      if ((err1!=0) || (err2!=0)) {
        ARCANE_FATAL("ERROR CUDA MemAdvise FAILED");
      }
      */
      // TODO: le prefetch fait chuter les perfs ... 
      /*
      auto err1 = cudaMemPrefetchAsync (mvis.data(), mvis.sizeBytes(), 0, *s);
      auto err2 = cudaMemPrefetchAsync (cids.data(), cids.sizeBytes(), 0, *s);
      if ((err1!=0) || (err2!=0)) {
        ARCANE_FATAL("ERROR de prefetch CUDA");
      }
      */

      // TODO: utiliser cudaLaunchKernel() à la place.
      impl::doIndirectGPULambda <<<b,t,0,*s>>>(mvis,cids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA compiler");
#endif
    break;
// TODO: A tester...
/*
  case eExecutionPolicy::HIP:
#if defined(ARCANE_COMPILING_HIP)
    {
      launch_info.beginExecute();
      SmallSpan<const MatVarIndex> mvis(items.component()->variableIndexer()->matvarIndexes());
      SmallSpan<const Int32> cids(items.component()->variableIndexer()->localIds());
      // TODO: vérifier que l'arcane assert n'est pas tout le temps fait
      ARCANE_ASSERT(mvis.size() == cids.size(), "MatVarIndex and CellLocalId arrays have different size");
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
      auto& loop_func = impl::doIndirectGPULambda<ItemType,Lambda>;
      hipLaunchKernelGGL(loop_func,b,t,0,*s,mvis,cids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting HIP kernel execution but the kernel is not compiled with HIP compiler");
#endif
    break;
*/
  case eExecutionPolicy::Sequential:
    {
      launch_info.beginExecute();
      // TODO: A voir avec GG si un for range est acceptable
      for (auto i : items.itemsInternalView())
        func(EnvCellAccessor(i));
      // TODO: Faut il remplacer le code ci-dessus par celui ci-dessous pour avoir un comportement équivalent entre CPU et GPU ?
      /*
      SmallSpan<const MatVarIndex> mvis(items.component()->variableIndexer()->matvarIndexes());
      SmallSpan<const Int32> cids(items.component()->variableIndexer()->localIds());
      assert(mvis.size() == cids.size());
      for (int i(0); i<mvis.size(); ++i)
        func(EnvCellAccessor(mvis[i], (CellLocalId)cids[i]));
      */
    }
    break;
  case eExecutionPolicy::Thread:
    {
      launch_info.beginExecute();
      arcaneParallelForeach(items,
                            [&](EnvCellVectorView sub_items)
                            {
                              impl::_doIndirectThreadLambda(sub_items,func);
                            });
    }
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
  impl::_applyEnvCells(command,items,std::forward<Lambda>(func));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Equivalent de la classe ItemRunCommand pour les EnvCell
 */
class EnvCellRunCommand
{
 public:
  EnvCellRunCommand(RunCommand& command,const EnvCellVectorView& items)
  : m_command(command), m_items(items) {}

  RunCommand& m_command;
  EnvCellVectorView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellRunCommand
operator<<(RunCommand& command,const EnvCellVectorView& items);

EnvCellRunCommand
operator<<(RunCommand& command,IMeshEnvironment* env);

template<typename Lambda>
void operator<<(EnvCellRunCommand&& nr,const Lambda& f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}

template<typename Lambda>
void operator<<(EnvCellRunCommand& nr,const Lambda& f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
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
