// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataReader.cc                                       (C) 2000-2021 */
/*                                                                           */
/* Lecteur de IData en parallèle.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/ParallelDataReader.h"

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
// Dichotomie
// Les xx doivent etre croissants
template<typename T>
class Bissection
{
 public:
  static Integer locate(ConstArrayView<T> xx, T x)
  {
    Integer n = xx.size();
    if (x<xx[0] || x>xx[n-1])
      return (-1);
    if (x==xx[0])
      return 0;
    if (x==xx[n-1])
      return (n-1);
    Integer jl = 0;
    Integer ju = n;
    while (ju-jl > 1) {
      Integer jm = (ju+jl) >> 1;
      if (x >= xx[jm])
        jl = jm;
      else
        ju = jm;
    }
    return jl;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture parallèle.
 *
 * Une instance de cette classe est associée à un groupe du maillage.
 *
 * Pour pouvoir l'utiliser, chaque rang du IParallelMng doit spécifier:
 * - la liste des uid qu'il souhaite, à remplir dans wantedUniqueIds()
 * - la liste triée par ordre croissant des uids qui sont gérés par ce rang, à remplir
 * dans writtenUniqueIds().
 * Une fois ceci fait, il faut appeler la méthode sort() pour calculer
 * les infos dont on a besoin pour l'envoie et la réception des valeurs.
 *
 * L'instance est alors utilisable pour toutes les variables qui reposent
 * sur ce groupe et il faut appeler getSortedValues() pour récupérer
 * les valeurs pour une variable.
 * 
 */
class ParallelDataReader::Impl
: public TraceAccessor
{
 public:
  Impl(IParallelMng* pm);
 public:

  Int64Array& writtenUniqueIds() { return m_written_unique_ids; }
  Int64Array& wantedUniqueIds() { return m_wanted_unique_ids; }

 private:

  IParallelMng* m_parallel_mng;

  Int32UniqueArray m_data_to_send_ranks;
  //TODO ne pas utiliser un tableau dimensionné au commSize()
  UniqueArray< SharedArray<Int32> > m_data_to_send_local_indexes;
  UniqueArray< SharedArray<Int32> > m_data_to_recv_indexes;
  Int64UniqueArray m_written_unique_ids;
  Int64UniqueArray m_wanted_unique_ids;
  Int32UniqueArray m_local_send_indexes;

 public:

 public:
  void sort();
 public:
  void getSortedValues(IData* written_data,IData* data);
 private:
  void _searchUniqueIdIndexes(Int64ConstArrayView recv_uids,
                              Int64ConstArrayView written_unique_ids,
                              Int32Array& indexes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataReader::
ParallelDataReader(IParallelMng* pm)
: m_p(new Impl(pm))
{
}

ParallelDataReader::
~ParallelDataReader()
{
  delete m_p;
}

Int64Array& ParallelDataReader::writtenUniqueIds()
{
  return m_p->writtenUniqueIds();
}
Int64Array& ParallelDataReader::
wantedUniqueIds()
{
  return m_p->wantedUniqueIds();
}
void ParallelDataReader::sort()
{
  return m_p->sort();
}
void ParallelDataReader::getSortedValues(IData* written_data,IData* data)
{
  m_p->getSortedValues(written_data,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataReader::Impl::
Impl(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataReader::Impl::
sort()
{
  Integer nb_wanted_uid = m_wanted_unique_ids.size();
    
  Integer nb_rank = m_parallel_mng->commSize();
  Integer my_rank = m_parallel_mng->commRank();

  Int64UniqueArray global_min_max_uid(nb_rank*2);
  {
    Int64 min_max_written_uid[2];
    min_max_written_uid[0] = NULL_ITEM_UNIQUE_ID;
    min_max_written_uid[1] = NULL_ITEM_UNIQUE_ID;
    Integer nb_written_uid = m_written_unique_ids.size();
      
    if (nb_written_uid!=0){
      // Les uid écrits sont triés par ordre croissant.
      // Le plus petit est donc le premier et le plus grand le dernier
      min_max_written_uid[0] = m_written_unique_ids[0];
      min_max_written_uid[1] = m_written_unique_ids[nb_written_uid-1];
    }
    m_parallel_mng->allGather(Int64ConstArrayView(2,min_max_written_uid),global_min_max_uid);
  }
  for( Integer irank=0; irank<nb_rank; ++irank )
    info(5) << "MIN_MAX_UIDS p=" << irank << " min=" << global_min_max_uid[irank*2]
            << " max=" << global_min_max_uid[(irank*2)+1];

  m_data_to_recv_indexes.resize(nb_rank);
  //Int32UniqueArray senders_rank(nb_rank);
  //senders_rank.fill(-1);
  {
    UniqueArray< SharedArray<Int64> > uids_list(nb_rank);
    //Integer current_sender = 0;
    auto exchanger { ParallelMngUtils::createExchangerRef(m_parallel_mng) };
    for( Integer i=0; i<nb_wanted_uid; ++i ){
      Int64 uid = m_wanted_unique_ids[i];
      Int32 rank = -1;
      //TODO: utiliser une dichotomie
      for( Int32 irank=0; irank<nb_rank; ++irank ){
        if (uid>=global_min_max_uid[irank*2] && uid<=global_min_max_uid[(irank*2)+1]){
          rank = irank;
          break;
        }
      }
      if (rank==(-1))
        fatal() << "Bad rank uid=" << uid;

      // Il est inutile de s'envoyer les valeurs
      if (rank!=my_rank){
        if (uids_list[rank].empty()){
          exchanger->addSender(rank);
          //senders_rank[rank] = current_sender;
          //uids_list.add(Int64UniqueArray());
        }
        uids_list[rank].add(uid);
      }
      m_data_to_recv_indexes[rank].add(i);
    }
    exchanger->initializeCommunicationsMessages();
    //info() << "NB SEND=" << exchanger->nbSender()
    //       << " NB_RECV=" << exchanger->nbReceiver();
    Int32ConstArrayView senders = exchanger->senderRanks();
    Integer nb_send = senders.size();
    for( Integer i=0; i<nb_send; ++i ){
      //info() << "READ SEND TO A: rank=" << senders[i];
      ISerializeMessage* send_msg = exchanger->messageToSend(i);
      Int32 dest_rank = senders[i];
      ISerializer* serializer = send_msg->serializer();
      //Integer nb_to_send = uids_list[dest_rank].size();
      //indexes_to_recv[i] = own_indexes_list[dest_rank]; //indexes_list[dest_rank];
      //ranks_to_recv[i] = dest_rank;
      serializer->setMode(ISerializer::ModeReserve);
      //serializer->reserveInteger(1);
      //serializer->reserve(DT_Int32,nb_to_send);
      serializer->reserveArray(uids_list[dest_rank]);
      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);
      //serializer->putInteger(nb_to_send);
      //serializer->put(indexes_list[dest_rank]);
      serializer->putArray(uids_list[dest_rank]);
#if 0
      for( Integer z=0; z<nb_to_send; ++z ){
        Integer index = indexes_list[dest_rank][z];
        info() << " SEND Z=" << z << " RANK=" << dest_rank << " index=" << index
               << " own_index=" << indexes_to_recv[i][z];
      }
#endif
    }
    exchanger->processExchange();
    Int32ConstArrayView receivers = exchanger->receiverRanks();
    Integer nb_recv = receivers.size();

    m_data_to_send_local_indexes.resize(nb_recv);
    m_data_to_send_ranks.resize(nb_recv);

    for( Integer i=0; i<nb_recv; ++i ){
      //info() << "READ RECEIVE FROM A: rank=" << receivers[i];
      ISerializeMessage* recv_msg = exchanger->messageToReceive(i);
      Int32 orig_rank = receivers[i];
      m_data_to_send_ranks[i] = orig_rank;
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      Int64UniqueArray recv_uids;
      serializer->getArray(recv_uids);
      Int64 nb_to_recv = recv_uids.largeSize();
      //Int32ArrayView own_group_local_ids = own_group.internal()->itemsLocalId();
      //info() << " RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv << " S2=" << own_group_local_ids.size();
      //for( Integer z=0; z<nb_to_recv; ++z ){
      //Integer index = recv_indexes[z];
      //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
      //     << " index2=" << own_group_local_ids[index];
      //recv_indexes[z] = own_group_local_ids[index];
      //}
      //info() << "READ END RECEIVE FROM A: NB_TO_RECEIVE " << nb_to_recv;

      m_data_to_send_local_indexes[i].resize(nb_to_recv);
      _searchUniqueIdIndexes(recv_uids,m_written_unique_ids,m_data_to_send_local_indexes[i]);
    }
  }

  // Traite les données qui sont déjà présentes sur ce processeur.
  {
    Integer my_rank = m_parallel_mng->commRank();
    Int32Array& local_recv_indexes = m_data_to_recv_indexes[my_rank];
    Integer nb_local_index = local_recv_indexes.size();
    if (nb_local_index>0){
      m_local_send_indexes.resize(nb_local_index);
      Int64UniqueArray uids(nb_local_index);
      for( Integer i=0; i<nb_local_index; ++i ){
        uids[i] = m_wanted_unique_ids[local_recv_indexes[i]];
      }
      _searchUniqueIdIndexes(uids,m_written_unique_ids,m_local_send_indexes);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataReader::Impl::
getSortedValues(IData* written_data,IData* data)
{
  auto exchanger { ParallelMngUtils::createExchangerRef(m_parallel_mng) };
  Integer nb_send = m_data_to_send_ranks.size();
  for( Integer i=0; i<nb_send; ++i ){
    exchanger->addSender(m_data_to_send_ranks[i]);
  }
  //TODO utiliser version sans allGather() car on connait les receptionneurs
  exchanger->initializeCommunicationsMessages();

  for( Integer i=0; i<nb_send; ++i ){
    ISerializeMessage* send_msg = exchanger->messageToSend(i);
    //info() << " SEND TO B: rank=" << send_msg->destSubDomain();
    //Int32 dest_rank = send_sd[i];
    ISerializer* serializer = send_msg->serializer();
    serializer->setMode(ISerializer::ModeReserve);
    if (written_data)
      written_data->serialize(serializer,m_data_to_send_local_indexes[i],0);
    serializer->allocateBuffer();
    serializer->setMode(ISerializer::ModePut);
    if (written_data)
      written_data->serialize(serializer,m_data_to_send_local_indexes[i],0);
  }
  exchanger->processExchange();

  Integer nb_wanted_uid = m_wanted_unique_ids.size();
  data->resize(nb_wanted_uid);

  // Traite les données qui sont déjà présente sur ce processeur.
  {
    Integer my_rank = m_parallel_mng->commRank();
    Int32Array& local_recv_indexes = m_data_to_recv_indexes[my_rank];
    Integer nb_local_index = local_recv_indexes.size();
    if (nb_local_index>0){
      //Int32UniqueArray local_send_indexes(nb_local_index);
      //Int64UniqueArray uids(nb_local_index);

      //for( Integer i=0; i<nb_local_index; ++i ){
      //uids[i] = m_wanted_unique_ids[local_recv_indexes[i]];
      //}
      //_searchUniqueIdIndexes(uids,m_written_unique_ids,local_send_indexes);

      //info() << "SERIALIZE RESERVE";
      SerializeBuffer sbuf;
      sbuf.setMode(ISerializer::ModeReserve);
      if (written_data)
        written_data->serialize(&sbuf,m_local_send_indexes,0);
      sbuf.allocateBuffer();
      //info() << "SERIALIZE PUT";
      sbuf.setMode(ISerializer::ModePut);
      if (written_data)
        written_data->serialize(&sbuf,m_local_send_indexes,0);
      //info() << "SERIALIZE GET";
      sbuf.setMode(ISerializer::ModeGet);
      data->serialize(&sbuf,local_recv_indexes,0);
    }
  }

  Int32ConstArrayView receivers = exchanger->receiverRanks();
  Integer nb_recv = receivers.size();
  for( Integer i=0; i<nb_recv; ++i ){
    ISerializeMessage* recv_msg = exchanger->messageToReceive(i);
    Int32 orig_rank = receivers[i];
    //info() << " RECEIVE FROM B: rank=" << orig_rank;
    ISerializer* serializer = recv_msg->serializer();
    serializer->setMode(ISerializer::ModeGet);
    data->serialize(serializer,m_data_to_recv_indexes[orig_rank],0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataReader::Impl::
_searchUniqueIdIndexes(Int64ConstArrayView recv_uids,
                       Int64ConstArrayView written_unique_ids,
                       Int32Array& indexes)
{
  Integer nb_to_recv = recv_uids.size();
  Integer nb_written_uid = written_unique_ids.size();

  for( Integer irecv=0; irecv<nb_to_recv; ++irecv ){
    //Integer my_index = -1;
    Int64 my_uid = recv_uids[irecv];
    // Comme les writtent_unique_ids sont triés, on peut utiliser une dichotomie
    Integer my_index = Bissection<Int64>::locate(written_unique_ids,my_uid);
    //info() << "MY_INDEX=" << my_index << " my_uid=" << my_uid;
    if (my_index==(-1))
      fatal() << "Can not find uid uid=" << my_uid << " (index=" << my_index << ")";
    //info() << "MY_INDEX2=" << my_index << " my_uid=" << my_uid;
    // Teste si la dichotomie est correcte
    if (written_unique_ids[my_index]!=my_uid)
      fatal() << "INTERNAL: bad index for bissection "
              << "Index=" << my_index << " uid=" << my_uid
              << " wuid=" << written_unique_ids[my_index]
              << " n=" << nb_written_uid;
    //info() << "Index=" << my_index << " uid=" << my_uid << " n=" << nb_written_uid;
    indexes[irecv] = my_index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
