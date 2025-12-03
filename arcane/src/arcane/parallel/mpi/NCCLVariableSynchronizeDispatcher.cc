// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NCCLVariableSynchronizeDispatcher.cc                        (C) 2000-2025 */
/*                                                                           */
/* Gestion spécifique des synchronisations des variables via NCCL.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/parallel/IStat.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/impl/IDataSynchronizeImplementation.h"

#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/accelerator/AcceleratorUtils.h"

#include <nccl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace Accelerator::Cuda
{
  void arcaneCheckNCCLErrors(const TraceInfo& ti, ncclResult_t e)
  {
    ARCCORE_FATAL_IF((e != ncclSuccess), "NCCL Error trace={0} e={1} str={2}", ti, e, ncclGetErrorString(e));
  }
} // namespace Accelerator::Cuda

#define ARCCORE_CHECK_NCCL(result) \
  Arcane::Accelerator::Cuda::arcaneCheckNCCLErrors(A_FUNCINFO, result)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation NCCL de la synchronisation.
 *
 * NCCL ne supporte qu'un seul GPU par rang.
 */
class NCCLVariableSynchronizeDispatcher
: public AbstractDataSynchronizeImplementation
{
 public:

  class Factory;
  explicit NCCLVariableSynchronizeDispatcher(Factory* f);

 protected:

  void compute() override {}
  void beginSynchronize(IDataSynchronizeBuffer* ds_buf) override;
  void endSynchronize(IDataSynchronizeBuffer* ds_buf) override;

 private:

  IParallelMng* m_parallel_mng = nullptr;
  ncclComm_t m_nccl_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NCCLVariableSynchronizeDispatcher::Factory
: public IDataSynchronizeImplementationFactory
{
 public:

  explicit Factory(IParallelMng* mpi_pm)
  : m_parallel_mng(mpi_pm)
  {}

  Ref<IDataSynchronizeImplementation> createInstance() override
  {
    auto* x = new NCCLVariableSynchronizeDispatcher(this);
    return makeRef<IDataSynchronizeImplementation>(x);
  }

 public:

  IParallelMng* m_parallel_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateNCCLVariableSynchronizerFactory(IParallelMng* mpi_pm)
{
  auto* x = new NCCLVariableSynchronizeDispatcher::Factory(mpi_pm);
  return makeRef<IDataSynchronizeImplementationFactory>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NCCLVariableSynchronizeDispatcher::
NCCLVariableSynchronizeDispatcher(Factory* f)
: m_parallel_mng(f->m_parallel_mng)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  // TODO: Il faudrait vérifier qu'on a bien un GPU par rang MPI
  // car NCCL ne supporte pas qu'il y ait plusieurs rangs sur le même GPU.

  ncclUniqueId my_id;
  ARCCORE_CHECK_NCCL(ncclGetUniqueId(&my_id));
  ArrayView<char> id_as_bytes(NCCL_UNIQUE_ID_BYTES, reinterpret_cast<char*>(&my_id));
  pm->broadcast(id_as_bytes, 0);

  ARCCORE_CHECK_NCCL(ncclCommInitRank(&m_nccl_communicator, nb_rank, my_id, my_rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NCCLVariableSynchronizeDispatcher::
beginSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  Integer nb_message = ds_buf->nbRank();

  IParallelMng* pm = m_parallel_mng;
  ITraceMng* tm = pm->traceMng();
  tm->info() << "Doing NCCL Sync";

  double prepare_time = 0.0;
  cudaStream_t stream = 0;

  // Si le IParallelMng a une RunQueue Cuda, on l'utilise.
  RunQueue pm_queue = pm->_internalApi()->queue();
  if (pm_queue.executionPolicy() == Accelerator::eExecutionPolicy::CUDA)
    stream = Accelerator::AcceleratorUtils::toCudaNativeStream(pm_queue);
;
  ARCCORE_CHECK_NCCL(ncclGroupStart());
  {
    // Recopie les buffers d'envoi dans \a var_values
    ds_buf->copyAllSend();

    // Poste les messages de réception
    for (Integer i = 0; i < nb_message; ++i) {
      Int32 target_rank = ds_buf->targetRank(i);
      auto buf = ds_buf->receiveBuffer(i).bytes();
      if (!buf.empty()) {
        ARCCORE_CHECK_NCCL(ncclRecv(buf.data(), buf.size(), ncclInt8, target_rank, m_nccl_communicator, stream));
      }
    }

    // Poste les messages d'envoi
    for (Integer i = 0; i < nb_message; ++i) {
      auto buf = ds_buf->sendBuffer(i).bytes();
      Int32 target_rank = ds_buf->targetRank(i);
      if (!buf.empty()) {
        ARCCORE_CHECK_NCCL(ncclSend(buf.data(), buf.size(), ncclInt8, target_rank, m_nccl_communicator, stream));
      }
    }
  }
  // Bloque jusqu'à ce que tous les messages soient terminés
  ARCCORE_CHECK_NCCL(ncclGroupEnd());

  tm->info() << "End begin synchronize";
  pm->stat()->add("SyncPrepare", prepare_time, ds_buf->totalSendSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NCCLVariableSynchronizeDispatcher::
endSynchronize(IDataSynchronizeBuffer* ds_buf)
{
  IParallelMng* pm = m_parallel_mng;

  double copy_time = 0.0;
  double wait_time = 0.0;
  ds_buf->copyAllReceive();

  // S'assure que les copies des buffers sont bien terminées
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
