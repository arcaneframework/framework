// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiDirectSendrecvVariableSynchronizeDispatcher.cc           (C) 2000-2025 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IDataSynchronizeImplementation.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de la synchronisation via MPI_Sendrecv.
 */
class MpiDirectSendrecvVariableSynchronizerDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit MpiDirectSendrecvVariableSynchronizerDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* vs_buf) override;
  void endSynchronize(IDataSynchronizeBuffer*) override
  {
    // With this implementation, we do not need this function.
  }

 private:

  MpiParallelMng* m_mpi_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiDirectSendrecvVariableSynchronizerDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  Factory(MpiParallelMng* mpi_pm)
  : m_mpi_parallel_mng(mpi_pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new MpiDirectSendrecvVariableSynchronizerDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  MpiParallelMng* m_mpi_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateMpiDirectSendrecvVariableSynchronizerFactory(MpiParallelMng* mpi_pm)
{
  auto* x = new MpiDirectSendrecvVariableSynchronizerDispatcher::Factory(mpi_pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiDirectSendrecvVariableSynchronizerDispatcher::
MpiDirectSendrecvVariableSynchronizerDispatcher(Factory* f)
: m_mpi_parallel_mng(f->m_mpi_parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiDirectSendrecvVariableSynchronizerDispatcher::
beginSynchronize(IDataSynchronizeBuffer* vs_buf)
{
  Int32 nb_message = vs_buf->nbRank();

  constexpr int serialize_tag = 523;

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();
  const MPI_Datatype mpi_dt = dtlist->datatype(Byte())->datatype();

  double sync_copy_send_time = 0.0;
  double sync_copy_recv_time = 0.0;
  double sync_wait_time = 0.0;

  {
    MpiTimeInterval tit(&sync_copy_send_time);
    vs_buf->copyAllSend();
  }

  {
    MpiTimeInterval tit(&sync_wait_time);
    for( Integer i=0; i<nb_message; ++i ){
      Int32 target_rank = vs_buf->targetRank(i);
      auto rbuf = vs_buf->receiveBuffer(i).bytes().smallView();
      auto sbuf = vs_buf->sendBuffer(i).bytes().smallView();

      MPI_Sendrecv(sbuf.data(), sbuf.size(), mpi_dt, target_rank, serialize_tag,
                   rbuf.data(), rbuf.size(), mpi_dt, target_rank, serialize_tag,
                   pm->communicator(), MPI_STATUS_IGNORE);
    }
  }

  {
    MpiTimeInterval tit(&sync_copy_recv_time);
    vs_buf->copyAllReceive();
  }

  pm->stat()->add("SyncCopySend",sync_copy_send_time,1);
  pm->stat()->add("SyncWait",sync_wait_time,1);
  pm->stat()->add("SyncCopyRecv",sync_copy_recv_time,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
