// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataReaderWriter.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Lecteur/Ecrivain de IData en parallèle.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/ParallelDataWriter.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializer.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/IData.h"
#include "arcane/parallel/BitonicSortT.H"
#include "arcane/ParallelMngUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelDataWriter::Impl
: public TraceAccessor
{
 public:

  Impl(IParallelMng* pm);

 public:

  Int64ConstArrayView sortedUniqueIds() const;
  void setGatherAll(bool v);

 private:

  IParallelMng* m_parallel_mng;
  //! Tableau indiquant les rangs des process dont on recoit des infos
  Int32UniqueArray m_ranks_to_send;
  //! Tableau indiquant les rangs des process auxquels on envoie des infos
  Int32UniqueArray m_ranks_to_recv;
  //TODO ne pas utiliser un tableau dimensionné au commSize()
  UniqueArray< SharedArray<Int32> > m_indexes_to_send;
  UniqueArray< SharedArray<Int32> > m_indexes_to_recv;
  Integer m_nb_item;
  Int64UniqueArray m_sorted_unique_ids;

  SharedArray<Int32> m_local_indexes_to_send;
  SharedArray<Int32> m_local_indexes_to_recv;

  bool m_gather_all;

 public:

  void sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid);

  Ref<IData> getSortedValues(IData* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataWriter::
ParallelDataWriter(IParallelMng* pm)
: m_p(new Impl(pm))
{
}

ParallelDataWriter::
~ParallelDataWriter()
{
  delete m_p;
}

Int64ConstArrayView ParallelDataWriter::
sortedUniqueIds() const
{
  return m_p->sortedUniqueIds();
}
void ParallelDataWriter::
setGatherAll(bool v)
{
  m_p->setGatherAll(v);
}

void ParallelDataWriter::
sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid)
{
  m_p->sort(local_ids,items_uid);
}

Ref<IData> ParallelDataWriter::
getSortedValues(IData* data)
{
  return m_p->getSortedValues(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataWriter::Impl::
Impl(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_nb_item(0)
, m_gather_all(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView ParallelDataWriter::Impl::
sortedUniqueIds() const
{
  return m_sorted_unique_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataWriter::Impl::
setGatherAll(bool v)
{
  m_gather_all = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataWriter::Impl::
sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid)
{
  //TODO: traiter les entites qui apres tri restent
  // sur le proc d'origine en utilisant une liste
  // speciale et ne pas envoyer/recevoir de messages
  IParallelMng* pm = m_parallel_mng;

  Parallel::BitonicSort<Int64> uid_sorter(pm);
  uid_sorter.sort(items_uid);

  Int32ConstArrayView key_indexes = uid_sorter.keyIndexes();
  Int32ConstArrayView key_ranks = uid_sorter.keyRanks();
  Int64ConstArrayView keys = uid_sorter.keys();

  Int64UniqueArray global_all_keys;
  Int32UniqueArray global_all_key_indexes;
  Int32UniqueArray global_all_key_ranks;

  Integer nb_item = keys.size();
  Int32 my_rank = pm->commRank();

#if 0
  for( Integer i=0; i<math::min(nb_item,20); ++i ){
    info() << "ORIGINAL I=" << i << " UID=" << items_uid[i]
           << " INDEX=" << key_indexes[i]
           << " RANK=" << key_ranks[i]
           << " KEY=" << keys[i];
  }
#endif

  if (m_gather_all){
    // Le proc 0 récupère tout
    pm->allGatherVariable(keys,global_all_keys);
    pm->allGatherVariable(key_indexes,global_all_key_indexes);
    pm->allGatherVariable(key_ranks,global_all_key_ranks);
    Int32 gather_rank = 0;

    if (pm->commRank()!=gather_rank){
      global_all_key_ranks.clear();
      global_all_key_indexes.clear();
      global_all_keys.clear();
    }
    nb_item = global_all_keys.size();
      
    key_ranks = global_all_key_ranks.view();
    key_indexes = global_all_key_indexes.view();
    keys = global_all_keys.view();
  }

  m_nb_item = nb_item;

  m_sorted_unique_ids.resize(nb_item);
  m_sorted_unique_ids.copy(keys);

  //info() << "END SORT SIZE=" << nb_item << " KEY_SIZE=" << keys.size();
#if 0
  for( Integer i=0; i<math::min(nb_item,20); ++i ){
    info() << "I=" << i << " KEY=" << keys[i]
           << " INDEX=" << key_indexes[i]
           << " RANK=" << key_ranks[i];
  }
#endif
  {
    UniqueArray< SharedArray<Int32> > indexes_list(pm->commSize());
    UniqueArray< SharedArray<Int32> > own_indexes_list(pm->commSize());
    //Int32UniqueArray rank_to_sends;
    auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };

    for( Integer i=0; i<nb_item; ++i ){
      Int32 index = key_indexes[i];
      Int32 rank = key_ranks[i];
      if(rank!=my_rank){
        if (indexes_list[rank].empty()){
          sd_exchange->addSender(rank);
        }
      }
      indexes_list[rank].add(index);
      own_indexes_list[rank].add(i);
    }
    m_local_indexes_to_recv = own_indexes_list[my_rank];
    m_local_indexes_to_send.resize(indexes_list[my_rank].size());

    //m_local_indexes_to_recv = indexes_list[my_rank];
    //m_local_indexes_to_send = own_indexes_list[my_rank];

    sd_exchange->initializeCommunicationsMessages();
    //info() << "NB SEND=" << sd_exchange->nbSender()
    //       << " NB_RECV=" << sd_exchange->nbReceiver();
    Int32ConstArrayView send_sd = sd_exchange->senderRanks();
    Integer nb_send = send_sd.size();
    m_indexes_to_recv.resize(nb_send);
    m_ranks_to_recv.resize(nb_send);
    for( Integer i=0; i<nb_send; ++i ){
      //info() << " SEND TO A: rank=" << send_sd[i];
      ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
      Int32 dest_rank = send_sd[i];
      ISerializer* serializer = send_msg->serializer();
      m_indexes_to_recv[i] = own_indexes_list[dest_rank]; //indexes_list[dest_rank];
      m_ranks_to_recv[i] = dest_rank;
      serializer->setMode(ISerializer::ModeReserve);

      serializer->reserveArray(indexes_list[dest_rank]);

      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);

      serializer->putArray(indexes_list[dest_rank]);
#if 0
      Integer nb_to_send = indexes_list[dest_rank].size();
      for( Integer z=0; z<nb_to_send; ++z ){
        Integer index = indexes_list[dest_rank][z];
        info() << " SEND Z=" << z << " RANK=" << dest_rank << " index=" << index
               << " own_index=" << indexes_to_recv[i][z];
      }
#endif
    }
    sd_exchange->processExchange();
    Int32ConstArrayView recv_sd = sd_exchange->receiverRanks();
    Integer nb_recv = recv_sd.size();
    m_indexes_to_send.resize(nb_recv);
    m_ranks_to_send.resize(nb_recv);
    for( Integer i=0; i<nb_recv; ++i ){
      //info() << " RECEIVE FROM A: rank=" << recv_sd[i];
      ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
      Int32 orig_rank = recv_sd[i];
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      //Integer nb_to_recv = serializer->getInteger();
      Int32Array& recv_indexes = m_indexes_to_send[i]; 
      m_ranks_to_send[i] = orig_rank;
      //recv_indexes.resize(nb_to_recv);
      serializer->getArray(recv_indexes);
      Int64 nb_to_recv = recv_indexes.largeSize();
      //Int32ArrayView own_group_local_ids = own_group.internal()->itemsLocalId();
      Int32ConstArrayView own_group_local_ids = local_ids;
      //info() << " RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv << " S2=" << own_group_local_ids.size();
      for( Integer z=0; z<nb_to_recv; ++z ){
        Integer index = recv_indexes[z];
        //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
        //     << " index2=" << own_group_local_ids[index];
        recv_indexes[z] = own_group_local_ids[index];
      }
      //info() << "END RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv;
    }

    // Traite les entités locales
    {
      Integer nb_local = m_local_indexes_to_send.size();
      for( Integer z=0; z<nb_local; ++z ){
        Integer index = indexes_list[my_rank][z];
        //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
        //     << " index2=" << own_group_local_ids[index];
        m_local_indexes_to_send[z] = local_ids[index];
      }
    }

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> ParallelDataWriter::Impl::
getSortedValues(IData* data)
{
  IParallelMng* pm = m_parallel_mng;
  Ref<IData> sorted_data = data->cloneEmptyRef();

  auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };
  for( Integer i=0, is=m_ranks_to_send.size(); i<is; ++i ){
    sd_exchange->addSender(m_ranks_to_send[i]);
  }
  Int32UniqueArray ranks_to_recv2;
  for( Integer i=0, is=m_ranks_to_recv.size(); i<is; ++i ){
    ranks_to_recv2.add(m_ranks_to_recv[i]);
  }
  sd_exchange->initializeCommunicationsMessages(ranks_to_recv2);
  Int32ConstArrayView send_sd = sd_exchange->senderRanks();
  Integer nb_send = send_sd.size();
  for( Integer i=0; i<nb_send; ++i ){
    //info() << " SEND TO B: rank=" << send_sd[i];
    ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
    //Int32 dest_rank = send_sd[i];
    ISerializer* serializer = send_msg->serializer();
    serializer->setMode(ISerializer::ModeReserve);
    data->serialize(serializer,m_indexes_to_send[i],0);
    serializer->allocateBuffer();
    serializer->setMode(ISerializer::ModePut);
    data->serialize(serializer,m_indexes_to_send[i],0);
  }
  sd_exchange->processExchange();
  Int32ConstArrayView recv_sd = sd_exchange->receiverRanks();
  Integer nb_recv = recv_sd.size();
  //m_indexes_to_send.resize(nb_recv);
  //ranks_to_send.resize(nb_recv);
  //IData* data2 = data;
  sorted_data->resize(m_nb_item);
  for( Integer i=0; i<nb_recv; ++i ){
    //info() << " RECEIVE FROM B: rank=" << recv_sd[i];
    ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
    //Int32 orig_rank = recv_sd[i];
    ISerializer* serializer = recv_msg->serializer();
    serializer->setMode(ISerializer::ModeGet);
    sorted_data->serialize(serializer,m_indexes_to_recv[i],0);
  }

  // Traite les données qui sont déjà présentes sur ce processeur.
  {
    //Integer my_rank = m_parallel_mng->commRank();
    Int32Array& local_recv_indexes = m_local_indexes_to_recv;
    Integer nb_local_index = local_recv_indexes.size();
    if (nb_local_index>0){
      SerializeBuffer sbuf;
      sbuf.setMode(ISerializer::ModeReserve);
      data->serialize(&sbuf,m_local_indexes_to_send,0);
      sbuf.allocateBuffer();
      sbuf.setMode(ISerializer::ModePut);
      data->serialize(&sbuf,m_local_indexes_to_send,0);
      sbuf.setMode(ISerializer::ModeGet);
      sorted_data->serialize(&sbuf,local_recv_indexes,0);
    }
  }
  return sorted_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
