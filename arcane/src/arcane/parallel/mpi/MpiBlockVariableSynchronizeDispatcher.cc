// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiBlockVariableSynchronizeDispatcher.cc                    (C) 2000-2022 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/MpiBlockVariableSynchronizeDispatcher.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpi/MpiAdapter.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiTimeInterval.h"
#include "arcane/parallel/IStat.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arccore/message_passing/IRequestList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette implémentation découpe la synchronisation en bloc de taille fixe.
 * Tout le mécanisme est dans _endSynchronize().
 * L'algorithme est le suivant:
 *
 * 1. Recopie dans les buffers d'envoi les valeurs à envoyer.
 * 2. Boucle sur Irecv/ISend/WaitAll tant que qu'il y a au moins une partie non vide.
 * 3. Recopie depuis les buffers de réception les valeurs des variables.
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType>
MpiBlockVariableSynchronizeDispatcher<SimpleType>::
MpiBlockVariableSynchronizeDispatcher(MpiBlockVariableSynchronizeDispatcherBuildInfo& bi)
: VariableSynchronizeDispatcher<SimpleType>(VariableSynchronizeDispatcherBuildInfo(bi.parallelMng(),bi.table()))
, m_mpi_parallel_mng(bi.parallelMng())
, m_request_list(m_mpi_parallel_mng->createRequestListRef())
, m_block_size(bi.blockSize())
, m_nb_sequence(bi.nbSequence())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> bool
MpiBlockVariableSynchronizeDispatcher<SimpleType>::
_isSkipRank(Int32 rank,Int32 sequence) const
{
  if (m_nb_sequence==1)
    return false;
  return (rank % m_nb_sequence) == sequence;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SimpleType> void
MpiBlockVariableSynchronizeDispatcher<SimpleType>::
_beginSynchronize(SyncBuffer& sync_buffer)
{
  // Ne fait rien au niveau MPI dans cette partie car cette implémentation
  // ne supporte pas l'asyncrhonisme.
  // On se contente de recopier les valeurs des variables dans le buffer d'envoi
  // pour permettre ensuite de modifier les valeurs de la variable entre
  // le _beginSynchronize() et le _endSynchronize().

  double send_copy_time = 0.0;
  {
    MpiTimeInterval tit(&send_copy_time);

    // Recopie les buffers d'envoi
    auto sync_list = this->m_sync_info->infos();
    Integer nb_message = sync_list.size();
    for (Integer i = 0; i < nb_message; ++i)
      sync_buffer.copySend(i);
  }
  Int64 total_share_size = sync_buffer.totalShareSize();
  m_mpi_parallel_mng->stat()->add("SyncSendCopy",send_copy_time,total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void
MpiBlockVariableSynchronizeDispatcher<SimpleType>::
_endSynchronize(SyncBuffer& sync_buffer)
{
  auto sync_list = this->m_sync_info->infos();
  const Int32 nb_message = sync_list.size();

  MpiParallelMng* pm = m_mpi_parallel_mng;
  MpiDatatypeList* dtlist = pm->datatypes();

  MP::Mpi::MpiAdapter* mpi_adapter = pm->adapter();

  double prepare_time = 0.0;
  double copy_time = 0.0;
  double wait_time = 0.0;

  constexpr int serialize_tag = 523;
  const MPI_Datatype mpi_dt = dtlist->datatype(SimpleType())->datatype();

  const Int32 block_size = m_block_size;

  for (Int32 isequence = 0; isequence<m_nb_sequence; ++isequence ){
    Int32 block_index = 0;
    while (1){
      {
        MpiTimeInterval tit(&prepare_time);
        m_request_list->clear();

        // Poste les messages de réception
        for (Integer i = 0; i < nb_message; ++i) {
          const VariableSyncInfo& vsi = sync_list[i];
          if (_isSkipRank(vsi.targetRank(),isequence))
            continue;
          ArrayView<SimpleType> buf0 = sync_buffer.ghostBuffer(i);
          ArrayView<SimpleType> buf = buf0.subView(block_index,block_size);
          if (!buf.empty()) {
            auto req = mpi_adapter->receiveNonBlockingNoStat(buf.data(), buf.size(),
                                                             vsi.targetRank(), mpi_dt, serialize_tag);
            m_request_list->add(req);
          }
        }

        // Poste les messages d'envoi en mode non bloquant.
        for (Integer i = 0; i < nb_message; ++i) {
          const VariableSyncInfo& vsi = sync_list[i];
          if (_isSkipRank(vsi.targetRank(),isequence))
            continue;
          ArrayView<SimpleType> buf0 = sync_buffer.shareBuffer(i);
          ArrayView<SimpleType> buf = buf0.subView(block_index,block_size);
          if (!buf.empty()) {
            auto request = mpi_adapter->sendNonBlockingNoStat(buf.data(), buf.size(),
                                                              vsi.targetRank(), mpi_dt, serialize_tag);
            m_request_list->add(request);
          }
        }
      }

      // Si aucune requête alors on a fini notre synchronisation
      if (m_request_list->size()==0)
        break;

      // Attend que les messages soient terminés
      {
        MpiTimeInterval tit(&wait_time);
        m_request_list->wait(Parallel::WaitAll);
      }

      block_index += block_size;
    }
  }


  // Recopie les valeurs recues
  {
    MpiTimeInterval tit(&copy_time);
    for (Integer i = 0; i < nb_message; ++i)
      sync_buffer.copyReceive(i);
  }

  Int64 total_ghost_size = sync_buffer.totalGhostSize();
  Int64 total_share_size = sync_buffer.totalShareSize();
  Int64 total_size = total_ghost_size + total_share_size;
  pm->stat()->add("SyncCopy",copy_time,total_ghost_size);
  pm->stat()->add("SyncWait",wait_time,total_size);
  pm->stat()->add("SyncPrepare", prepare_time, total_share_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiBlockVariableSynchronizeDispatcher<Byte>;
template class MpiBlockVariableSynchronizeDispatcher<Int16>;
template class MpiBlockVariableSynchronizeDispatcher<Int32>;
template class MpiBlockVariableSynchronizeDispatcher<Int64>;
template class MpiBlockVariableSynchronizeDispatcher<Real>;
template class MpiBlockVariableSynchronizeDispatcher<Real2>;
template class MpiBlockVariableSynchronizeDispatcher<Real3>;
template class MpiBlockVariableSynchronizeDispatcher<Real2x2>;
template class MpiBlockVariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
