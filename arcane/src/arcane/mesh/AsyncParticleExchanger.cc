// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncParticleExchanger.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Echangeur de particules asynchrone.                                       */
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
  // Par défaut met à 0 le niveau de verbosité pour éviter trop de messages
  // lors des phases asynchrones.
  m_bpe.setVerboseLevel(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AsyncParticleExchanger::
initialize(IItemFamily* item_family)
{
  m_bpe.initialize(item_family);
  IParallelMng* pm = m_bpe.m_parallel_mng;
  if (pm->isParallel()){
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

  //Génère tous les Isend et les Imrecv matchés avec Improbe
  m_bpe.m_nb_particle_send = local_ids.size();
  {
    Timer::Sentry ts(m_bpe.m_timer);
    _generateSendItemsAsync(local_ids, sub_domains_to_send);
  }
  if (m_bpe.m_verbose_level>=1)
    info() << "ASE_BeginLoop loop=" << m_bpe.m_nb_loop;
  m_bpe._sendPendingMessages();

  if (new_particle_local_ids)
    new_particle_local_ids->clear();

  bool has_new_particle = _waitSomeMessages(ItemGroup(), new_particle_local_ids);
  if (has_new_particle)
    has_local_flying_particles = true;

  //----------------------------------------
  //Ici on a le coeur de l'algo de condition d'arrêt lors de l'utilisation de AsyncParticleExchanger
  //
  //Si taille de chunk == 0 && pas de req(red) en vol
  //Si (Q > 0) avec Q le nombre de particule en vol (résultat du Iallreduce)
  //  Iallreduce (P, Q, req(red));
  //  P=0; avec P le nombre de particule envoyé depuis le dernier Iallreduce
  //Sinon
  //  retourner is_finished = true
  //

  IParallelMng* pm = m_bpe.m_parallel_mng;
  UniqueArray<Integer> isIallReduceRunning = pm->testSomeRequests(m_reduce_requests);

  //Si la requête a matché, on clear le tableau de requête
  if (isIallReduceRunning.size() != 0){
    m_reduce_requests.clear();
    if (m_bpe.m_verbose_level>=1)
      info() << "PSM_IAllReduceFinished loop=" << m_bpe.m_nb_loop
             << " total=" << m_sum_of_nb_particle_sent;
  }

  //Ici, on teste si on a des particules à traiter en local
  //Qu'il n'y a pas de requête Iallreduce en vol
  //et qu'il n'y a pas de requête à envoyer ou recevoir en vol
  if ((!has_local_flying_particles) && (m_reduce_requests.size() == 0) && (m_bpe.m_waiting_messages.size() == 0) && (m_bpe.m_pending_messages.size()==0)) {
    if (m_sum_of_nb_particle_sent > 0) {
      //Faire MPI_Iallreduce
      IParallelNonBlockingCollective* pnbc = pm->nonBlockingCollective();
      m_nb_particle_send_before_reduction = m_nb_particle_send_before_reduction_tmp;
      if (m_bpe.m_verbose_level>=1)
        info() << "PSM_DoIAllReduce loop=" << m_bpe.m_nb_loop
               << " n=" << m_nb_particle_send_before_reduction
               << " nb_to_send=" << local_ids.size();
      m_reduce_requests.add(pnbc->allReduce(Parallel::ReduceSum,
                                            ConstArrayView<Integer>(1, &m_nb_particle_send_before_reduction),
                                            ArrayView<Integer>(1, &m_sum_of_nb_particle_sent)));
      m_nb_particle_send_before_reduction_tmp = 0;
    }
    else {
      is_finished = true; // is_finished = true, il n'y a plus de particules à traiter globalement
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
  // Infos pour chaque sous-domaine connecté
  m_bpe.m_accumulate_infos.clear();
  m_bpe.m_accumulate_infos.resize(nb_connected_sub_domain);

  m_bpe._addItemsToSend(local_ids, sub_domains_to_send, communicating_sub_domains, ids_to_send);

  Int64UniqueArray items_to_send_uid;
  Int64UniqueArray items_to_send_cells_uid; // Uniquement pour les particules;

  IParallelMng* pm = m_bpe.m_parallel_mng;

  //-------------------------------
  // Gestion des envoies de particules
  //
  // [HT] En mode asynchrone, nous devons envoyer que si nous avons des particules
  // et les réceptions se feront avec des MPI_Improbe
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
  // Gestion des réceptions de particules
  //
  // [HT] En mode asynchrone, les réceptions se font avec des MPI_Improbe et MPI_Imrecv
  for (Integer j = 0; j < nb_connected_sub_domain; ++j) {

    MessageTag tag(Arcane::MessagePassing::internal::BasicSerializeMessage::DEFAULT_SERIALIZE_TAG_VALUE);
    MessageRank rank(communicating_sub_domains[j]);
    PointToPointMessageInfo message(rank,tag);
    message.setBlocking(false);
    MessageId mid = pm->probe(message);

    if (mid.isValid()) {
      SerializeMessage* recv_sm = new SerializeMessage(m_bpe.subDomain()->subDomainId(), mid);
      m_bpe.m_pending_messages.add(recv_sm);
    }
  }

  m_bpe.m_accumulate_infos.clear();
  // Détruit les entités qui viennent d'être envoyées
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

  // Sauve les communications actuellements traitées car le traitement
  // peut en ajouter de nouvelles
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
      if (!sm->isSend()) { //Si le msg est un recv
        m_bpe._deserializeMessage(sm, items_to_create_unique_id, items_to_create_cells_unique_id,
                                  items_to_create_local_id, items_to_create_cells_local_id,
                                  item_group, new_particle_local_ids);
        // Indique qu'on a recu des particules et donc il faudrait dire
        // que has_local_flying_particle est vra
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
