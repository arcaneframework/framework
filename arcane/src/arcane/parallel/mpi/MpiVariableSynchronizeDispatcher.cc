// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiVariableSynchronizeDispatcher.cc                         (C) 2000-2025 */
/*                                                                           */
/* Specific MPI handling for variable synchronization.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IDataSynchronizeImplementation.h"

#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * The synchronization algorithm works as follows. The first three points are
 * in beginSynchronize() and the last two are in endSynchronize(). The current
 * code only allows for non-blocking synchronization at a time.
 *
 * 1. Post receive messages
 * 2. Copy the values to be sent into the send buffers. This is done after
 *    posting the receive messages to allow for some overlap between computation
 *    and communication.
 * 3. Post send messages.
 * 4. Perform a WaitSome on the receive messages. As soon as a message
 *    arrives, the receive buffer is copied into the variable array. The code
 *    could be simplified by using WaitAll and copying all values at the end.
 * 5. Perform a WaitAll on the send messages to free the requests.
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Optimized implementation for MPI synchronization.
 *
 * Compared to the base version, this implementation uses MPI_Waitsome
 * (instead of Waitall) and copies into the destination buffer
 * as soon as a message arrives.
 *
 * NOTE: this optimization respects the MPI standard which states that
 * you must not touch the memory area of a message until it is complete.
 */
class MpiVariableSynchronizeDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit MpiVariableSynchronizeDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* ds_buf) override;
  void endSynchronize(IDataSynchronizeBuffer* ds_buf) override;

 private:

  MpiParallelMng* m_mpi_parallel_mng;
  UniqueArray<Parallel::Request> m_original_recv_requests;
  UniqueArray<bool> m_original_recv_requests_done;
  Ref<Parallel::IRequestList> m_receive_request_list;
  Ref<Parallel::IRequestList> m_send_request_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiVariableSynchronizeDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(MpiParallelMng* mpi_pm)
  : m_mpi_parallel_mng(mpi_pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new MpiVariableSynchronizeDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiVariableSynchronizerFactory(MpiParallelMng* mpi_pm)
{
  auto* x = new MpiVariableSynchronizeDispatcher::Factory(mpi_pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiVariableSynchronizeDispatcher::
MpiVariableSynchronizeDispatcher(Factory* f)
: m_mpi_parallel_mng(f->m_mpi_parallel_mng)
, m_receive_request_list(m_mpi_parallel_mng->createRequestListRef())
, m_send_request_list(m_mpi_parallel_mng->createRequestListRef())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiVariableSynchronizeDispatcher::
beginSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  Integer nb_message = ds_buf->nbRank();

  m_send_request_list->clear();

  MpiParallelMng* pm = m_mpi_parallel_mng;

  MP::Mpi::MpiAdapter* mpi_adapter = pm->adapter();
  const MPI_Datatype mpi_dt = MP::Mpi::MpiBuiltIn::datatype(Byte());

  double prepare_time = 0.0;

  {
    MpiTimeInterval tit(&prepare_time);
    constexpr int serialize_tag = 523;

    // Send receive messages in non-blocking mode
    m_original_recv_requests_done.resize(nb_message);
    m_original_recv_requests.resize(nb_message);

    // Post receive messages
    for (Integer i = 0; i < nb_message; ++i) {
      Int32 target_rank = ds_buf->targetRank(i);
      auto buf = ds_buf->receiveBuffer(i).bytes();
      if (!buf.empty()) {
        auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(), buf.size(),
                                                         target_rank, mpi_dt, serialize_tag);
        m_original_recv_requests[i] = req;
        m_original_recv_requests_done[i] = false;
      }
      else {
        // It is not necessary to send an empty message.
        // Consider the message finished
        m_original_recv_requests[i] = Parallel::Request{};
        m_original_recv_requests_done[i] = true;
      }
    }

    // Copy send buffers into \a var_values
    ds_buf->copyAllSend();

    // Post send messages in non-blocking mode.
    for (Integer i = 0; i < nb_message; ++i) {
      auto buf = ds_buf->sendBuffer(i).bytes();
      Int32 target_rank = ds_buf->targetRank(i);
      if (!buf.empty()) {
        auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(), buf.size(),
                                                          target_rank, mpi_dt, serialize_tag);
        m_send_request_list->add(request);
      }
    }
  }
  pm->stat()->add("SyncPrepare", prepare_time, ds_buf->totalSendSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiVariableSynchronizeDispatcher::
endSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  MpiParallelMng* pm = m_mpi_parallel_mng;

  // We need to keep the original index in 'SyncBuffer'
  // of each request to manage the copies.
  UniqueArray<Integer> remaining_original_indexes;

  double copy_time = 0.0;
  double wait_time = 0.0;

  while (1) {
    // Create the list of still active requests.
    m_receive_request_list->clear();
    remaining_original_indexes.clear();
    for (Integer i = 0, n = m_original_recv_requests_done.size(); i < n; ++i) {
      if (!m_original_recv_requests_done[i]) {
        m_receive_request_list->add(m_original_recv_requests[i]);
        remaining_original_indexes.add(i);
      }
    }
    Integer nb_remaining_request = m_receive_request_list->size();
    if (nb_remaining_request == 0)
      break;

    {
      MpiTimeInterval tit(&wait_time);
      m_receive_request_list->wait(Parallel::WaitSome);
    }

    // For each completed request, perform the copy
    ConstArrayView<Int32> done_requests = m_receive_request_list->doneRequestIndexes();

    for (Int32 request_index : done_requests) {
      Int32 orig_index = remaining_original_indexes[request_index];

      // To indicate that it is finished
      m_original_recv_requests_done[orig_index] = true;

      // Copy the received values
      {
        MpiTimeInterval tit(&copy_time);
        ds_buf->copyReceiveAsync(orig_index);
      }
    }
  }

  // Wait for sends to finish.
  // This must be done to free the requests even if the message
  // has arrived.
  {
    MpiTimeInterval tit(&wait_time);
    m_send_request_list->wait(Parallel::WaitAll);
  }

  // Ensure that buffer copies are completed
  ds_buf->barrier();

  Int64 total_ghost_size = ds_buf->totalReceiveSize();
  Int64 total_share_size = ds_buf->totalSendSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy", copy_time, total_ghost_size);
  pm->stat()->add("SyncWait", wait_time, total_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
