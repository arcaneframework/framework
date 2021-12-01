// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.cc                           (C) 2000-2021 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/VariableSynchronizerDispatcher.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/VariableCollection.h"
#include "arcane/ParallelMngUtils.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/ISerializer.h"
#include "arcane/IData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
applyDispatch(IArrayDataT<SimpleType>* data)
{
  this->beginSynchronize(data->view(),m_1d_buffer);
  this->endSynchronize(data->view(),m_1d_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
  
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
_copyFromBuffer(Int32ConstArrayView indexes,ConstArrayView<SimpleType> buffer,
                ArrayView<SimpleType> var_value,Integer dim2_size)
{
  if (dim2_size==1)
    m_buffer_copier->copyFromBufferOne(indexes,buffer,var_value);
  else
    m_buffer_copier->copyFromBufferMultiple(indexes,buffer,var_value,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
_copyToBuffer(Int32ConstArrayView indexes,ArrayView<SimpleType> buffer,
              ConstArrayView<SimpleType> var_value,Integer dim2_size)
{
  if (dim2_size==1)
    m_buffer_copier->copyToBufferOne(indexes,buffer,var_value);
  else
    m_buffer_copier->copyToBufferMultiple(indexes,buffer,var_value,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
applyDispatch(IArray2DataT<SimpleType>* data)
{
  Array2View<SimpleType> value = data->view();
  SimpleType* value_ptr = value.data();
  // Cette valeur doit être la même sur tous les procs
  Integer dim2_size = value.dim2Size();
  if (dim2_size==0)
    return;
  Integer dim1_size = value.dim1Size();
  m_2d_buffer.compute(m_sync_list,dim2_size);
  ArrayView<SimpleType> buf(dim1_size*dim2_size,value_ptr);
  this->beginSynchronize(buf,m_2d_buffer);
  this->endSynchronize(buf,m_2d_buffer);
  //TODO: liberer la memoire si besoin ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 * \todo: ne pas allouer les tampons car leur conservation est couteuse en
 * terme de memoire.
 */
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::SyncBuffer::
compute(ConstArrayView<VariableSyncInfo> sync_list,Integer dim2_size)
{
  m_dim2_size = dim2_size;
  Integer nb_message = sync_list.size();

  m_ghost_locals_buffer.resize(nb_message);
  m_share_locals_buffer.resize(nb_message);

  Integer total_ghost_buffer = 0;
  Integer total_share_buffer = 0;
  for( Integer i=0; i<nb_message; ++i ){
    total_ghost_buffer += sync_list[i].nbGhost();
    total_share_buffer += sync_list[i].nbShare();
  }
  m_ghost_buffer.resize(total_ghost_buffer*dim2_size);
  m_share_buffer.resize(total_share_buffer*dim2_size);
    
  {
    Integer array_index = 0;
    for( Integer i=0, is=sync_list.size(); i<is; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      Int32ConstArrayView ghost_grp = vsi.ghostIds();
      Integer local_size = ghost_grp.size();
      m_ghost_locals_buffer[i] = ArrayView<SimpleType>();
      if (local_size!=0)
        m_ghost_locals_buffer[i] = ArrayView<SimpleType>(local_size*dim2_size,
                                                         &m_ghost_buffer[array_index*dim2_size]);
      array_index += local_size;
    }
  }
  {
    Integer array_index = 0;
    for( Integer i=0, is=sync_list.size(); i<is; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      Int32ConstArrayView share_grp = vsi.shareIds();
      Integer local_size = share_grp.size();
      m_share_locals_buffer[i] = ArrayView<SimpleType>();
      if (local_size!=0)
        m_share_locals_buffer[i] = ArrayView<SimpleType>(local_size*dim2_size,
                                                         &m_share_buffer[array_index*dim2_size]);
      array_index += local_size;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 */
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
compute(ConstArrayView<VariableSyncInfo> sync_list)
{
  //IParallelMng* pm = m_parallel_mng;
  m_sync_list = sync_list;
  //Integer nb_message = sync_list.size();
  //pm->traceMng()->info() << "** RECOMPUTE SYNC LIST!!! N=" << nb_message
  //                       << " this=" << (IVariableSynchronizeDispatcher*)this
  //                       << " m_sync_list=" << &m_sync_list;

  m_1d_buffer.compute(sync_list,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
beginSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (m_is_in_sync)
    ARCANE_FATAL("Only one pending serialisation is supported");
  
  IParallelMng* pm = m_parallel_mng;
  
  //ITraceMng* trace = pm->traceMng();
  //Integer nb_elem = var_values.size();
  bool use_blocking_send = false;
  Integer nb_message = m_sync_list.size();
  Integer dim2_size = sync_buffer.m_dim2_size;

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  //SyncBuffer& sync_buffer = m_1d_buffer;
  // Envoie les messages de réception non bloquant
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
    if (!ghost_local_buffer.empty()){
      Parallel::Request rval = pm->recv(ghost_local_buffer,vsi.targetRank(),false);
      m_all_requests.add(rval);
    }
  }

  // Envoie les messages d'envoie en mode non bloquant.
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    Int32ConstArrayView share_grp = vsi.shareIds();
    ArrayView<SimpleType> share_local_buffer = sync_buffer.m_share_locals_buffer[i];
      
    _copyToBuffer(share_grp,share_local_buffer,var_values,dim2_size);

    ConstArrayView<SimpleType> const_share = share_local_buffer;
    if (!share_local_buffer.empty()){
      //for( Integer i=0, is=share_local_buffer.size(); i<is; ++i )
      //trace->info() << "TO rank=" << vsi.m_target_rank << " I=" << i << " V=" << share_local_buffer[i]
      //                << " lid=" << share_grp[i] << " v2=" << var_values[share_grp[i]];
      Parallel::Request rval = pm->send(const_share,vsi.targetRank(),use_blocking_send);
      if (!use_blocking_send)
        m_all_requests.add(rval);
    }
  }
  m_is_in_sync = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
endSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (!m_is_in_sync)
    ARCANE_FATAL("endSynchronize() called but no beginSynchronize() was called before");

  IParallelMng* pm = m_parallel_mng;
  
  //ITraceMng* trace = pm->traceMng();
  //Integer nb_elem = var_values.size();
  //bool use_blocking_send = false;
  Integer nb_message = m_sync_list.size();
  Integer dim2_size = sync_buffer.m_dim2_size;

  /*pm->traceMng()->info() << " ** ** COMMON END SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/


  // Attend que les receptions se terminent
  pm->waitAllRequests(m_all_requests);
  m_all_requests.clear();

  // Recopie dans la variable le message de retour.
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    Int32ConstArrayView ghost_grp = vsi.ghostIds();
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
    _copyFromBuffer(ghost_grp,ghost_local_buffer,var_values,dim2_size);
    //for( Integer i=0, is=ghost_local_buffer.size(); i<is; ++i )
    //trace->info() << "RECV rank=" << vsi.m_target_rank << " I=" << i << " V=" << ghost_local_buffer[i]
    //                << " lid=" << ghost_grp[i] << " v2=" << var_values[ghost_grp[i]];
  }

  m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerMultiDispatcher::
synchronize(VariableCollection vars,ConstArrayView<VariableSyncInfo> sync_infos)
{
  Ref<IParallelExchanger> exchanger{ParallelMngUtils::createExchangerRef(m_parallel_mng)};
  Integer nb_rank = sync_infos.size();
  Int32UniqueArray recv_ranks(nb_rank);
  for( Integer i=0; i<nb_rank; ++i ){
    Int32 rank = sync_infos[i].targetRank();
    exchanger->addSender(rank);
    recv_ranks[i] = rank;
  }
  exchanger->initializeCommunicationsMessages(recv_ranks);
  for( Integer i=0; i<nb_rank; ++i ){
    ISerializeMessage* msg = exchanger->messageToSend(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView share_ids = sync_infos[i].shareIds();
    sbuf->setMode(ISerializer::ModeReserve);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->serialize(sbuf,share_ids,0);
    }
    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->serialize(sbuf,share_ids,0);
    }
  }
  exchanger->processExchange();
  for( Integer i=0; i<nb_rank; ++i ){
    ISerializeMessage* msg = exchanger->messageToReceive(i);
    ISerializer* sbuf = msg->serializer();
    Int32ConstArrayView ghost_ids = sync_infos[i].ghostIds();
    sbuf->setMode(ISerializer::ModeGet);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->serialize(sbuf,ghost_ids,0);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerDispatcher::
~VariableSynchronizerDispatcher()
{
  delete m_dispatcher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerDispatcher::
compute(ConstArrayView<VariableSyncInfo> sync_list)
{
  ConstArrayView<IVariableSynchronizeDispatcher*> dispatchers = m_dispatcher->dispatchers();
  m_parallel_mng->traceMng()->info(4) << "DISPATCH RECOMPUTE";
  for( Integer i=0, is=dispatchers.size(); i<is; ++i )
    dispatchers[i]->compute(sync_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
_changeIds(Array<Int32>& ids,Int32ConstArrayView old_to_new_ids)
{
  UniqueArray<Int32> orig_ids(ids);
  ids.clear();

  for( Integer z=0, zs=orig_ids.size(); z<zs; ++z ){
    Int32 old_id = orig_ids[z];
    Int32 new_id = old_to_new_ids[old_id];
    if (new_id!=NULL_ITEM_LOCAL_ID)
      ids.add(new_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  _changeIds(m_share_ids,old_to_new_ids);
  _changeIds(m_ghost_ids,old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class VariableSynchronizeDispatcher<Byte>;
template class VariableSynchronizeDispatcher<Real>;
template class VariableSynchronizeDispatcher<Int16>;
template class VariableSynchronizeDispatcher<Int32>;
template class VariableSynchronizeDispatcher<Int64>;
template class VariableSynchronizeDispatcher<Real2>;
template class VariableSynchronizeDispatcher<Real3>;
template class VariableSynchronizeDispatcher<Real2x2>;
template class VariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
