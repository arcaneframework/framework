// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataReader.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Parallel IData Reader.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/ParallelDataReader.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IData.h"
#include "arcane/core/parallel/BitonicSortT.H"
#include "arcane/core/ParallelMngUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parallel reading.
 *
 * An instance of this class is associated with a mesh group.
 *
 * To use it, each rank of IParallelMng must specify:
 * - the list of uids it wants, to be filled in wantedUniqueIds()
 * - the list sorted in ascending order of uids managed by this rank, to be filled
 * in writtenUniqueIds().
 * Once this is done, the sort() method must be called to calculate
 * the information needed for sending and receiving values.
 *
 * The instance is then usable for all variables that rely
 * on this group, and getSortedValues() must be called to retrieve
 * the values for a variable.
 */
class ParallelDataReader::Impl
: public TraceAccessor
{
 public:

  explicit Impl(IParallelMng* pm);

 public:

  Int64Array& writtenUniqueIds() { return m_written_unique_ids; }
  Int64Array& wantedUniqueIds() { return m_wanted_unique_ids; }

 private:

  IParallelMng* m_parallel_mng = nullptr;

  Int32UniqueArray m_data_to_send_ranks;
  //TODO do not use an array sized to commSize()
  UniqueArray<SharedArray<Int32>> m_data_to_send_local_indexes;
  UniqueArray<SharedArray<Int32>> m_data_to_recv_indexes;
  Int64UniqueArray m_written_unique_ids;
  Int64UniqueArray m_wanted_unique_ids;
  Int32UniqueArray m_local_send_indexes;

 public:

  void sort();

 public:

  void getSortedValues(IData* written_data, IData* data);

 private:

  void _searchUniqueIdIndexes(Int64ConstArrayView recv_uids,
                              Int64ConstArrayView written_unique_ids,
                              Int32Array& indexes) const;
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

Array<Int64>& ParallelDataReader::
writtenUniqueIds()
{
  return m_p->writtenUniqueIds();
}
Array<Int64>& ParallelDataReader::
wantedUniqueIds()
{
  return m_p->wantedUniqueIds();
}
void ParallelDataReader::
sort()
{
  return m_p->sort();
}
void ParallelDataReader::
getSortedValues(IData* written_data, IData* data)
{
  m_p->getSortedValues(written_data, data);
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
  Int32 nb_wanted_uid = m_wanted_unique_ids.size();

  Int32 nb_rank = m_parallel_mng->commSize();
  Int32 my_rank = m_parallel_mng->commRank();

  Int64UniqueArray global_min_max_uid(nb_rank * 2);
  {
    FixedArray<Int64, 2> min_max_written_uid;
    min_max_written_uid[0] = NULL_ITEM_UNIQUE_ID;
    min_max_written_uid[1] = NULL_ITEM_UNIQUE_ID;
    Integer nb_written_uid = m_written_unique_ids.size();

    if (nb_written_uid != 0) {
      // The written uids are sorted in ascending order.
      // The smallest is therefore the first and the largest the last
      min_max_written_uid[0] = m_written_unique_ids[0];
      min_max_written_uid[1] = m_written_unique_ids[nb_written_uid - 1];
    }
    m_parallel_mng->allGather(min_max_written_uid.view(), global_min_max_uid);
  }
  for (Integer irank = 0; irank < nb_rank; ++irank)
    info(5) << "MIN_MAX_UIDS p=" << irank << " min=" << global_min_max_uid[irank * 2]
            << " max=" << global_min_max_uid[(irank * 2) + 1];

  m_data_to_recv_indexes.resize(nb_rank);
  {
    UniqueArray<SharedArray<Int64>> uids_list(nb_rank);
    auto exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };
    for (Integer i = 0; i < nb_wanted_uid; ++i) {
      Int64 uid = m_wanted_unique_ids[i];
      Int32 rank = -1;
      //TODO: use a dichotomy
      for (Int32 irank = 0; irank < nb_rank; ++irank) {
        if (uid >= global_min_max_uid[irank * 2] && uid <= global_min_max_uid[(irank * 2) + 1]) {
          rank = irank;
          break;
        }
      }
      if (rank == (-1))
        ARCANE_FATAL("Bad rank uid={0} uid_index={1}", uid, i);

      // It is unnecessary to send the values
      if (rank != my_rank) {
        if (uids_list[rank].empty()) {
          exchanger->addSender(rank);
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
    for (Integer i = 0; i < nb_send; ++i) {
      //info() << "READ SEND TO A: rank=" << senders[i];
      ISerializeMessage* send_msg = exchanger->messageToSend(i);
      Int32 dest_rank = senders[i];
      ISerializer* serializer = send_msg->serializer();
      serializer->setMode(ISerializer::ModeReserve);
      serializer->reserveArray(uids_list[dest_rank]);
      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);
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

    for (Integer i = 0; i < nb_recv; ++i) {
      //info() << "READ RECEIVE FROM A: rank=" << receivers[i];
      ISerializeMessage* recv_msg = exchanger->messageToReceive(i);
      Int32 orig_rank = receivers[i];
      m_data_to_send_ranks[i] = orig_rank;
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      Int64UniqueArray recv_uids;
      serializer->getArray(recv_uids);
      Int64 nb_to_recv = recv_uids.largeSize();

      m_data_to_send_local_indexes[i].resize(nb_to_recv);
      _searchUniqueIdIndexes(recv_uids, m_written_unique_ids, m_data_to_send_local_indexes[i]);
    }
  }

  // Processes the data that is already present on this processor.
  {
    Int32Array& local_recv_indexes = m_data_to_recv_indexes[my_rank];
    Integer nb_local_index = local_recv_indexes.size();
    if (nb_local_index > 0) {
      m_local_send_indexes.resize(nb_local_index);
      Int64UniqueArray uids(nb_local_index);
      for (Integer i = 0; i < nb_local_index; ++i) {
        uids[i] = m_wanted_unique_ids[local_recv_indexes[i]];
      }
      _searchUniqueIdIndexes(uids, m_written_unique_ids, m_local_send_indexes);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataReader::Impl::
getSortedValues(IData* written_data, IData* data)
{
  auto exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };
  Integer nb_send = m_data_to_send_ranks.size();
  for (Integer i = 0; i < nb_send; ++i) {
    exchanger->addSender(m_data_to_send_ranks[i]);
  }
  //TODO use version without allGather() since we know the receivers
  exchanger->initializeCommunicationsMessages();

  for (Integer i = 0; i < nb_send; ++i) {
    ISerializeMessage* send_msg = exchanger->messageToSend(i);
    //info() << " SEND TO B: rank=" << send_msg->destSubDomain();
    //Int32 dest_rank = send_sd[i];
    ISerializer* serializer = send_msg->serializer();
    serializer->setMode(ISerializer::ModeReserve);
    if (written_data)
      written_data->serialize(serializer, m_data_to_send_local_indexes[i], 0);
    serializer->allocateBuffer();
    serializer->setMode(ISerializer::ModePut);
    if (written_data)
      written_data->serialize(serializer, m_data_to_send_local_indexes[i], 0);
  }
  exchanger->processExchange();

  Integer nb_wanted_uid = m_wanted_unique_ids.size();
  data->resize(nb_wanted_uid);

  // Processes the data that is already present on this processor.
  {
    Integer my_rank = m_parallel_mng->commRank();
    ConstArrayView<Int32> local_recv_indexes = m_data_to_recv_indexes[my_rank];
    Integer nb_local_index = local_recv_indexes.size();
    if (nb_local_index > 0) {
      //info() << "SERIALIZE RESERVE";
      SerializeBuffer sbuf;
      sbuf.setMode(ISerializer::ModeReserve);
      if (written_data)
        written_data->serialize(&sbuf, m_local_send_indexes, 0);
      sbuf.allocateBuffer();
      //info() << "SERIALIZE PUT";
      sbuf.setMode(ISerializer::ModePut);
      if (written_data)
        written_data->serialize(&sbuf, m_local_send_indexes, 0);
      //info() << "SERIALIZE GET";
      sbuf.setMode(ISerializer::ModeGet);
      data->serialize(&sbuf, local_recv_indexes, 0);
    }
  }

  Int32ConstArrayView receivers = exchanger->receiverRanks();
  Integer nb_recv = receivers.size();
  for (Integer i = 0; i < nb_recv; ++i) {
    ISerializeMessage* recv_msg = exchanger->messageToReceive(i);
    Int32 orig_rank = receivers[i];
    //info() << " RECEIVE FROM B: rank=" << orig_rank;
    ISerializer* serializer = recv_msg->serializer();
    serializer->setMode(ISerializer::ModeGet);
    data->serialize(serializer, m_data_to_recv_indexes[orig_rank], 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelDataReader::Impl::
_searchUniqueIdIndexes(Int64ConstArrayView recv_uids,
                       Int64ConstArrayView written_unique_ids,
                       Array<Int32>& indexes) const
{
  Integer nb_to_recv = recv_uids.size();
  Integer nb_written_uid = written_unique_ids.size();

  for (Integer irecv = 0; irecv < nb_to_recv; ++irecv) {
    Int64 my_uid = recv_uids[irecv];
    // Since written_unique_ids are sorted, we can use a dichotomy
    auto iter_end = written_unique_ids.end();
    auto iter_begin = written_unique_ids.begin();
    auto x2 = std::lower_bound(iter_begin, iter_end, my_uid);
    if (x2 == iter_end)
      ARCANE_FATAL("Can not find uid uid={0} (with binary_search)", my_uid);
    Int32 my_index = CheckedConvert::toInt32(x2 - iter_begin);

    // Test if the dichotomy is correct
    if (written_unique_ids[my_index] != my_uid)
      ARCANE_FATAL("INTERNAL: bad index for bissection "
                   "Index={0} uid={1} wuid={2} n={3}",
                   my_index, my_uid, written_unique_ids[my_index], nb_written_uid);

    //info() << "Index=" << my_index << " uid=" << my_uid << " n=" << nb_written_uid;
    indexes[irecv] = my_index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
