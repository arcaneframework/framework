// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandEnumerate.h                                       (C) 2000-2021 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une liste d'entités.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunQueueInternal.h"

#include "arcane/ItemTypes.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename Lambda>
void _doIndirectThreadLambda(ItemVectorViewT<ItemType> sub_items,Lambda func)
{
  typedef typename ItemType::LocalIdType LocalIdType;

  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  ENUMERATE_ITEM(iitem,sub_items){
    body(LocalIdType(iitem.itemLocalId()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 */
template<typename ItemType,typename Lambda> void
_applyItems(RunCommand& command,ItemVectorViewT<ItemType> items,Lambda func)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Integer vsize = items.size();
  if (vsize==0)
    return;
  typedef typename ItemType::LocalIdType LocalIdType;
  impl::RunCommandLaunchInfo launch_info(command);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
#if defined(ARCANE_COMPILING_CUDA)
    {
      launch_info.beginExecute();
      Span<const Int32> local_ids = items.localIds();
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
      // TODO: utiliser cudaLaunchKernel() à la place.
      impl::doIndirectCUDALambda<ItemType,Lambda> <<<b,t,0,*s>>>(local_ids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA compiler");
#endif
    break;
  case eExecutionPolicy::HIP:
#if defined(__HIP__)
    {
      launch_info.beginExecute();
      Span<const Int32> local_ids = items.localIds();
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
      auto& loop_func = impl::doIndirectCUDALambda<ItemType,Lambda>;
      hipLaunchKernelGGL(loop_func,b,t,0,*s, local_ids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting HIP kernel execution but the kernel is not compiled with HIP compiler");
#endif

  case eExecutionPolicy::Sequential:
    {
      launch_info.beginExecute();
      ENUMERATE_ITEM(iitem,items){
        func(LocalIdType(iitem.itemLocalId()));
      }
    }
    break;
  case eExecutionPolicy::Thread:
    {
      launch_info.beginExecute();
      arcaneParallelForeach(items,
                            [&](ItemVectorViewT<ItemType> sub_items)
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

//! Applique la lambda \a func sur l'intervalle d'itération donnée par \a bounds
template<typename ItemType,typename Lambda> void
run(RunCommand& command,const ItemGroupT<ItemType>& items,Lambda func)
{
  impl::_applyItems<ItemType>(command,items.view(),std::forward<Lambda>(func));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename Lambda> void
run(RunCommand& command,ItemVectorViewT<ItemType> items,Lambda func)
{
  impl::_applyItems<ItemType>(command,items,std::forward<Lambda>(func));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType>
class ItemRunCommand
{
 public:
  ItemRunCommand(RunCommand& command,const ItemVectorViewT<ItemType>& items)
  : m_command(command), m_items(items)
  {
  }
  RunCommand& m_command;
  ItemVectorViewT<ItemType> m_items;
};

template<typename ItemType> ItemRunCommand<ItemType>
operator<<(RunCommand& command,const ItemGroupT<ItemType>& items)
{
  return ItemRunCommand<ItemType>(command,items.view());
}

template<typename ItemType> ItemRunCommand<ItemType>
operator<<(RunCommand& command,const ItemVectorViewT<ItemType>& items)
{
  return ItemRunCommand<ItemType>(command,items);
}

template<typename ItemType,typename Lambda>
void operator<<(ItemRunCommand<ItemType>&& nr,Lambda f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}
template<typename ItemType,typename Lambda>
void operator<<(ItemRunCommand<ItemType>& nr,Lambda f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer un groupe d'entités
#define RUNCOMMAND_ENUMERATE(ItemNameType,iter_name,item_group)         \
  A_FUNCINFO << item_group << [=] ARCCORE_HOST_DEVICE (ItemNameType##LocalId iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
