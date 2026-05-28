// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncParticleExchanger.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Asynchronous particle exchanger.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/AsyncParticleExchanger.h"
#include "arcane/core/IParallelNonBlockingCollective.h"

#include "arcane/core/internal/SerializeMessage.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
using namespace Arcane::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AsyncParticleExchanger::
AsyncParticleExchanger(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_bpe(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AsyncParticleExchanger::
~AsyncParticleExchanger()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
build()
{
  m_bpe.build();
  // By default sets the verbosity level to 0 to avoid too many messages
  // during asynchronous phases.
  m_bpe.setVerboseLevel(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
initialize(IItemFamily* item_family)
{
  m_bpe.initialize(item_family);
  IParallelMng* pm = m_bpe.m_parallel_mng;
  if (pm->isParallel()) {
    IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
    if (!pnbc)
      ARCANE_THROW(NotSupportedException,
                   "AsyncParticleExchanger is not supported because NonBlocking"
                   " collectives are not available");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
beginNewExchange(Integer nb_particule)
{
  info() << "AsyncParticleExchanger is used. It also use BasicParticleExchanger functionnalities";
  m_bpe.beginNewExchange(nb_particule);

  m_nb_particle_send_before_reduction = 0;
  m_nb_particle_send_before_reduction_tmp = 0;
  m_sum_of_nb_particle_sent = 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AsyncParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send,
              ItemGroup item_group,
              IFunctor* functor)
{
  return m_bpe.exchangeItems(nb_particle_finish_exchange, local_ids,
                             sub_domains_to_send, item_group, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AsyncParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send,
              Int32Array* new_particle_local_ids,
              IFunctor* functor)
{
  return m_bpe.exchangeItems(nb_particle_finish_exchange, local_ids,
                             sub_domains_to_send, new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
sendItems(Integer nb_particle_finish_exchange,
          Int32ConstArrayView local_ids,
          Int32ConstArrayView sub_domains_to_send)
{
  m_bpe.sendItems(nb_particle_finish_exchange, local_ids, sub_domains_to_send);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AsyncParticleExchanger::
waitMessages(Integer nb_pending_particles,
             Int32Array* new_particle_local_ids,
             IFunctor* functor)
{
  return m_bpe.waitMessages(nb_pending_particles, new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
addNewParticles(Integer nb_particle)
{
  m_bpe.addNewParticles(nb_particle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* AsyncParticleExchanger::
itemFamily()
{
  return m_bpe.itemFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
setVerboseLevel(Integer level)
{
  m_bpe.setVerboseLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer AsyncParticleExchanger::
verboseLevel() const
{
  return m_bpe.verboseLevel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAsyncParticleExchanger* AsyncParticleExchanger::
asyncParticleExchanger()
{
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AsyncParticleExchanger::
exchangeItemsAsync(Integer nb_particle_finish_exchange,
                   Int32ConstArrayView local_ids,
                   Int32ConstArrayView sub_domains_to_send,
                   Int32Array* new_particle_local_ids,
                   IFunctor* functor,
                   bool has_local_flying_particles)
{
  ARCANE_UNUSED(nb_particle_finish_exchange);
  ARCANE_UNUSED(functor);

  bool is_finished = false;
  ++m_bpe.m_nb_loop;

  // Generates all Isend and Imrecv matched with Improbe
  m_bpe.m_nb_particle_send = local_ids.size();
  {
    Timer::Sentry ts(m_bpe.m_timer);
    _generateSendItemsAsync(local_ids, sub_domains_to_send);
  }
  if (m_bpe.m_verbose_level >= 1)
    info() << "ASE_BeginLoop loop=" << m_bpe.m_nb_loop;
  m_bpe._sendPendingMessages();

  if (new_particle_local_ids)
    new_particle_local_ids->clear();

  bool has_new_particle = _waitSomeMessages(ItemGroup(), new_particle_local_ids);
  if (has_new_particle)
    has_local_flying_particles = true;

  //----------------------------------------
  // Here is the core of the stopping condition algorithm when using AsyncParticleExchanger
  //
  //If chunk size == 0 && no req(red) in flight
  //If (Q > 0) with Q being the number of particles in flight (result of the Iallreduce)
  //  Iallreduce (P, Q, req(red));
  //  P=0; with P being the number of particles sent since the last Iallreduce
  //Otherwise
  //  return is_finished = true
  //

  IParallelMng* pm = m_bpe.m_parallel_mng;
  UniqueArray<Integer> isIallReduceRunning = pm->testSomeRequests(m_reduce_requests);

  //If the request matched, we clear the request array
  if (isIallReduceRunning.size() != 0) {
    m_reduce_requests.clear();
    if (m_bpe.m_verbose_level >= 1)
      info() << "PSM_IAllReduceFinished loop=" << m_bpe.m_nb_loop
             << " total=" << m_sum_of_nb_particle_sent;
  }

  //Here, we test if we have particles to process locally
  //If there are no Iallreduce requests in flight
  //and no requests to send or receive in flight
  if ((!has_local_flying_particles) && (m_reduce_requests.size() == 0) && (m_bpe.m_waiting_messages.size() == 0) && (m_bpe.m_pending_messages.size() == 0)) {
    if (m_sum_of_nb_particle_sent > 0) {
      //Perform MPI_Iallreduce
      IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
      m_nb_particle_send_before_reduction = m_nb_particle_send_before_reduction_tmp;
      if (m_bpe.m_verbose_level >= 1)
        info() << "PSM_DoIAllReduce loop=" << m_bpe.m_nb_loop
               << " n=" << m_nb_particle_send_before_reduction
               << " nb_to_send=" << local_ids.size();
      m_reduce_requests.add(pnbc->allReduce(Parallel::ReduceSum,
                                            ConstArrayView<Integer>(1, &m_nb_particle_send_before_reduction),
                                            ArrayView<Integer>(1, &m_sum_of_nb_particle_sent)));
      m_nb_particle_send_before_reduction_tmp = 0;
    }
    else {
      is_finished = true; // is_finished = true, there are no more particles to process globally
    }
  }
  return is_finished;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
_generateSendItemsAsync(Int32ConstArrayView local_ids, Int32ConstArrayView sub_domains_to_send)
{
  Timer::Phase tphase(subDomain(), TP_Communication);

  IMesh* mesh = m_bpe.m_item_family->mesh();

  Int32UniqueArray communicating_sub_domains;
  mesh->cellFamily()->getCommunicatingSubDomains(communicating_sub_domains);

  Integer nb_connected_sub_domain = communicating_sub_domains.size();

  UniqueArray<SharedArray<Int32>> ids_to_send(nb_connected_sub_domain);
  // Info for each connected sub-domain
  m_bpe.m_accumulate_infos.clear();
  m_bpe.m_accumulate_infos.resize(nb_connected_sub_domain);

  m_bpe._addItemsToSend(local_ids, sub_domains_to_send, communicating_sub_domains, ids_to_send);

  Int64UniqueArray items_to_send_uid;
  Int64UniqueArray items_to_send_cells_uid; // Only for particles;

  IParallelMng* pm = m_bpe.m_parallel_mng;

  //-------------------------------
  // Handling particle sends
  //
  // [HT] In asynchronous mode, we must only send if we have particles
  // and receptions will be done with MPI_Improbe
  for (Integer j = 0; j < nb_connected_sub_domain; ++j) {
    if (ids_to_send[j].size() != 0) {
      auto* sm = new SerializeMessage(pm->commRank(), communicating_sub_domains[j],
                                      ISerializeMessage::MT_Send);
      m_bpe.m_accumulate_infos[j] = sm;
      m_bpe._serializeMessage(sm, ids_to_send[j], items_to_send_uid, items_to_send_cells_uid);
      m_bpe.m_pending_messages.add(sm);
      m_nb_particle_send_before_reduction_tmp += ids_to_send[j].size();
    }
  }

  //-------------------------------
  // Handling particle receives
  //
  // [HT] In asynchronous mode, receptions are done with MPI_Improbe and MPI_Imrecv
  for (Integer j = 0; j < nb_connected_sub_domain; ++j) {

    MessageTag tag(Arcane::MessagePassing::internal::BasicSerializeMessage::DEFAULT_SERIALIZE_TAG_VALUE);
    MessageRank rank(communicating_sub_domains[j]);
    PointToPointMessageInfo message(rank, tag);
    message.setBlocking(false);
    MessageId mid = pm->probe(message);

    if (mid.isValid()) {
      SerializeMessage* recv_sm = new SerializeMessage(m_bpe.subDomain()->subDomainId(), mid);
      m_bpe.m_pending_messages.add(recv_sm);
    }
  }

  m_bpe.m_accumulate_infos.clear();
  // Deletes the entities that were just sent
  m_bpe.m_item_family->toParticleFamily()->removeParticles(local_ids);
  m_bpe.m_item_family->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AsyncParticleExchanger::
_waitSomeMessages(ItemGroup item_group, Int32Array* new_particle_local_ids)
{
  {
    Timer::Sentry ts(m_bpe.m_timer);
    m_bpe.m_message_list->waitMessages(Parallel::WaitSomeNonBlocking);
  }
  m_bpe.m_total_time_waiting += m_bpe.m_timer->lastActivationTime();

  // Save the currently processed communications because processing
  // might add new ones
  UniqueArray<ISerializeMessage*> current_messages(m_bpe.m_waiting_messages);

  m_bpe.m_waiting_messages.clear();

  Int64UniqueArray items_to_create_unique_id;
  Int64UniqueArray items_to_create_cells_unique_id;
  Int32UniqueArray items_to_create_local_id;
  Int32UniqueArray items_to_create_cells_local_id;
  bool has_new_particle = false;
  for (Integer i = 0, is = current_messages.size(); i < is; ++i) {
    ISerializeMessage* sm = current_messages[i];
    if (sm->finished()) {
      if (!sm->isSend()) { //If the msg is a recv
        m_bpe._deserializeMessage(sm, items_to_create_unique_id, items_to_create_cells_unique_id,
                                  items_to_create_local_id, items_to_create_cells_local_id,
                                  item_group, new_particle_local_ids);
        // Indicates that particles were received and therefore it should be stated
        // that has_local_flying_particle is true
        if (!items_to_create_unique_id.empty())
          has_new_particle = true;
      }
      delete sm;
    }
    else {
      m_bpe.m_waiting_messages.add(sm);
    }
  }
  return has_new_particle;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AsyncParticleExchanger,
                                           IParticleExchanger,
                                           AsyncParticleExchanger);
ARCANE_REGISTER_SUB_DOMAIN_FACTORY(AsyncParticleExchanger,
                                   IParticleExchanger,
                                   AsyncParticleExchanger);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
