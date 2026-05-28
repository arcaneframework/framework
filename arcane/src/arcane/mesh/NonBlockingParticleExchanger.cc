// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NonBlockingParticleExchanger.cc                             (C) 2000-2025 */
/*                                                                           */
/* Particle Exchanger.                                                       */
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/NonBlockingParticleExchanger.h"

#include "arcane/utils/List.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Item.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/internal/SerializeMessage.h"

//#define ARCANE_DEBUG_EXCHANGE_ITEMS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NonBlockingParticleExchanger::
NonBlockingParticleExchanger(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_item_family(nullptr)
, m_parallel_mng(sbi.mesh()->parallelMng())
, m_rank(m_parallel_mng->commRank())
, m_timer(new Timer(m_parallel_mng->timerMng(), "NonBlockingParticleExchanger", Timer::TimerReal))
, m_total_time_functor(0.)
, m_total_time_waiting(0.)
, m_nb_total_particle_finish_exchange(0)
, m_nb_total_particle(0)
, m_nb_original_blocking_size(0)
, m_nb_blocking_size(m_nb_original_blocking_size)
, m_exchange_finished(true)
, m_master_proc(0)
, m_need_general_receive(false)
, m_end_message_sended(false)
, m_can_process_messages(true)
, m_can_process_non_blocking(false)
, m_want_process_non_blocking(false)
, m_want_fast_send_particles(true)
, m_nb_receive_message(0)
, m_nb_particle_finished_exchange(0)
, m_verbose_level(1)
, m_is_debug(false)
{
// m_want_fast_send_particles allows sending the number of particles that have finished tracking
// at the same time, avoiding additional messages.
#if ARCANE_DEBUG_EXCHANGE_ITEMS
  m_is_debug = true;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NonBlockingParticleExchanger::
~NonBlockingParticleExchanger()
{
  // No throws are allowed in the destructor, so compilation warnings are generated
  if (!m_pending_messages.empty() || !m_waiting_messages.empty()) {
    String s = String::format("pending or waiting messages: nb_pending={0} nb_waiting=",
                              m_pending_messages.size(), m_waiting_messages.size());
    warning() << s;
  }

  if (!m_waiting_local_ids.empty()) {
    warning() << String::format("pending particles: nb_pending=", m_waiting_local_ids.size());
  }

  delete m_timer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
initialize(IItemFamily* item_family)
{
  m_item_family = item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_clearMessages()
{
  for (Integer i = 0, is = m_accumulate_infos.size(); i < is; ++i) {
    delete m_accumulate_infos[i];
    m_accumulate_infos[i] = 0;
  }
  m_accumulate_infos.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
beginNewExchange(Integer i_nb_particle)
{
  _checkInitialized();
  IParallelMng* pm = m_parallel_mng;

  m_end_message_sended = false;
  m_exchange_finished = false;
  m_nb_blocking_size = m_nb_original_blocking_size;
  m_nb_receive_message = 0;
  m_nb_particle_finished_exchange = 0;

  // TODO: Use a specific tag for this exchange.
  Int64 nb_particle = i_nb_particle;
  m_nb_total_particle = pm->reduce(Parallel::ReduceSum, nb_particle);
  info() << "BEGIN TRACKING TOTAL FLYING = " << m_nb_total_particle
         << " (local=" << nb_particle << ") "
         << " (Date=" << platform::getCurrentDateTime() << ")";

  m_nb_total_particle_finish_exchange = 0;

  m_need_general_receive = true;

  // Retrieve the list of variables to transfer.
  // These are variables from the same family as the ones passed as parameters.
  // IMPORTANT: All sub-domains must have these same variables.
  m_variables_to_exchange.clear();
  m_item_family->usedVariables(m_variables_to_exchange);
  m_variables_to_exchange.sortByName(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NonBlockingParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send, ItemGroup item_group,
              IFunctor* functor)
{
  // TODO: Remove this if it's already handled by sendItems()
  m_nb_particle_finished_exchange += nb_particle_finish_exchange;
  return _exchangeItems(local_ids, sub_domains_to_send, item_group, 0, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NonBlockingParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send,
              Int32Array* new_particle_local_ids,
              IFunctor* functor)
{
  // TODO: Remove this if it's already handled by sendItems()
  m_nb_particle_finished_exchange += nb_particle_finish_exchange;
  return _exchangeItems(local_ids, sub_domains_to_send, ItemGroup(), new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
sendItems(Integer nb_particle_finish_exchange,
          Int32ConstArrayView local_ids,
          Int32ConstArrayView sub_domains_to_send)
{
  m_nb_particle_finished_exchange += nb_particle_finish_exchange;
  _checkSendItems(local_ids, sub_domains_to_send);
  _sendPendingMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NonBlockingParticleExchanger::
waitMessages(Integer nb_pending_particle, Int32Array* new_particle_local_ids, IFunctor* functor)
{
  return _waitMessages(nb_pending_particle, ItemGroup(), new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_checkSendItems(Int32ConstArrayView local_ids,
                Int32ConstArrayView sub_domains_to_send)
{
  Integer nb_particle = local_ids.size();
  Integer nb_waiting_local_ids = m_waiting_local_ids.size();
  Integer nb_waiting_sub_domains_to_send = m_waiting_sub_domains_to_send.size();
  if ((nb_particle + nb_waiting_local_ids) >= m_nb_blocking_size) {
    _generateSendItemsMessages(local_ids, sub_domains_to_send);
  }
  else {
    // Place the particles in a buffer before sending them
    m_waiting_local_ids.resize(nb_waiting_local_ids + nb_particle);
    m_waiting_sub_domains_to_send.resize(nb_waiting_sub_domains_to_send + nb_particle);
    for (Integer i = 0; i < nb_particle; ++i) {
      m_waiting_local_ids[nb_waiting_local_ids + i] = local_ids[i];
      m_waiting_sub_domains_to_send[nb_waiting_sub_domains_to_send + i] = sub_domains_to_send[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NonBlockingParticleExchanger::
_exchangeItems(Int32ConstArrayView local_ids,
               Int32ConstArrayView sub_domains_to_send, ItemGroup item_group,
               Int32Array* new_particle_local_ids,
               IFunctor* functor)
{
  if (m_want_fast_send_particles) {
    if (local_ids.empty())
      _processFinishTrackingMessage();
  }
  else if (!m_want_process_non_blocking)
    _processFinishTrackingMessage();
  if (m_exchange_finished) {
    _sendFinishExchangeParticle();
    m_need_general_receive = false;
  }
  _checkNeedReceiveMessage();

  _checkSendItems(local_ids, sub_domains_to_send);

  return _waitMessages(0, item_group, new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NonBlockingParticleExchanger::
_waitMessages(Integer nb_pending_particle, ItemGroup item_group,
              Int32Array* new_particle_local_ids, IFunctor* functor)
{
  ARCANE_UNUSED(nb_pending_particle);

  if (!item_group.null())
    item_group.clear();

  m_can_process_messages = true;
  m_can_process_non_blocking = m_want_process_non_blocking;
  while (m_can_process_messages && !m_exchange_finished) {
    m_can_process_messages = false;
    _processMessages(item_group, new_particle_local_ids, false, functor);
  }

  if (m_exchange_finished) {
    info(5) << " ** EXCHANGE finished: ";
    // This ensures that all completion messages are received
    _processMessages(item_group, new_particle_local_ids, true, 0);
    info(5) << " ** EXCHANGE finished END: ";
  }

  info(5) << " ** RETURN EXCHANGE m_exchange_finished: " << m_exchange_finished;
  return m_exchange_finished;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_checkNeedReceiveMessage()
{
  if (m_need_general_receive) {
    auto sm = new SerializeMessage(m_rank, A_NULL_RANK, ISerializeMessage::MT_Recv);
    m_pending_messages.add(sm);
    m_need_general_receive = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_generateSendItemsMessages(Int32ConstArrayView local_ids,
                           Int32ConstArrayView sub_domains_to_send)
{
  Timer::Phase tphase(m_parallel_mng->timeStats(), TP_Communication);

  IMesh* mesh = m_item_family->mesh();

  Int32UniqueArray communicating_sub_domains;
  mesh->cellFamily()->getCommunicatingSubDomains(communicating_sub_domains);

  Integer nb_connected_sub_domain = communicating_sub_domains.size();
  //Integer max_sub_domain_id = 0;
  UniqueArray<SharedArray<Int32>> ids_to_send(nb_connected_sub_domain);
  // Information for each connected sub-domain
  //_clearMessages();
  m_accumulate_infos.clear();
  m_accumulate_infos.resize(nb_connected_sub_domain);
  for (Integer i = 0; i < nb_connected_sub_domain; ++i)
    m_accumulate_infos[i] = new SerializeMessage(m_rank, communicating_sub_domains[i],
                                                 ISerializeMessage::MT_Send);

  _addItemsToSend(local_ids, sub_domains_to_send, communicating_sub_domains, ids_to_send);
  _addItemsToSend(m_waiting_local_ids, m_waiting_sub_domains_to_send,
                  communicating_sub_domains, ids_to_send);

  if (m_is_debug) {
    info() << "-- Subdomain " << m_rank << ". NB to send: " << local_ids.size()
           << " NB connected subdomains: " << nb_connected_sub_domain;

    info() << "NB connected subdomain for " << m_rank << " : " << m_accumulate_infos.size();
    for (Integer i = 0, n = m_accumulate_infos.size(); i < n; ++i) {
      info() << "------------- Send: rank=" << m_accumulate_infos[i]->destRank()
             << " n=" << ids_to_send[i].size();
    }
  }

  Int64UniqueArray items_to_send_uid;
  Int64UniqueArray items_to_send_cells_uid; // Only for particles;

  for (Integer j = 0; j < nb_connected_sub_domain; ++j) {
    ISerializeMessage* sm = m_accumulate_infos[j];
    // In blocking mode, always send the message because the recipient has already sent
    // a reception message. Otherwise, send it only if it contains particles.
    if (!ids_to_send[j].empty())
      _serializeMessage(sm, ids_to_send[j], items_to_send_uid,
                        items_to_send_cells_uid);
    else
      // The message is useless since it's empty.
      delete sm;
  }

  m_accumulate_infos.clear();

  // Destroy the entities that have just been sent
  info(5) << "NonBlockingParticleExchanger:: sendItems " << "local_ids           " << local_ids.size();
  info(5) << "NonBlockingParticleExchanger:: sendItems " << "m_waiting_local_ids " << m_waiting_local_ids.size();

  m_item_family->toParticleFamily()->removeParticles(local_ids);
  m_item_family->toParticleFamily()->removeParticles(m_waiting_local_ids);
  m_item_family->endUpdate();
  m_waiting_local_ids.clear();
  m_waiting_sub_domains_to_send.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_addItemsToSend(Int32ConstArrayView local_ids,
                Int32ConstArrayView sub_domains_to_send,
                Int32ConstArrayView communicating_sub_domains,
                UniqueArray<SharedArray<Int32>>& ids_to_send)
{
  String func_name("NonBlockingParticleExchanger::_addItemsToSend()");
  Integer nb_connected_sub_domain = ids_to_send.size();
  // Determine to which sub-domain each item should be sent.
  // This is done based on the local_ids parameter.
  Integer id_size = local_ids.size();
  for (Integer i = 0; i < id_size; ++i) {
    Int32 item_local_id = local_ids[i];
    Integer sd_to_send = sub_domains_to_send[i];
#ifdef ARCANE_CHECK
    if (sd_to_send == m_rank)
      // This item belongs to this sub-domain.
      fatal() << func_name << "The entity with local id " << item_local_id
              << " should not be sent to its own subdomain";
#endif
    // Find the index of the sub-domain to which the item belongs
    // in the list of connected sub-domains.
    // TODO: Use an indirect method (e.g., a table based on the number of sub-domains).
    Integer sd_index = nb_connected_sub_domain;
    for (Integer i_sd = 0; i_sd < nb_connected_sub_domain; ++i_sd)
      if (sd_to_send == communicating_sub_domains[i_sd]) {
        sd_index = i_sd;
        break;
      }
#ifdef ARCANE_CHECK
    if (sd_index == nb_connected_sub_domain)
      fatal() << func_name << "Internal: bad subdomain index";
#endif
    ids_to_send[sd_index].add(item_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_processMessages(ItemGroup item_group, Int32Array* new_particle_local_ids,
                 bool wait_all, IFunctor* functor)
{
  _sendPendingMessages();

  if (functor) {
    {
      Timer::Sentry ts(m_timer);
      functor->executeFunctor();
    }
    m_total_time_functor += m_timer->lastActivationTime();
    info(5) << "TimeFunctor: current=" << m_timer->lastActivationTime()
            << " total=" << m_total_time_functor;
  }

  Integer nb_message_finished = 0;
  {
    Timer::Sentry ts(m_timer);
    if (wait_all)
      nb_message_finished = m_message_list->waitMessages(Parallel::WaitAll);
    else {
      if (m_can_process_non_blocking)
        nb_message_finished = m_message_list->waitMessages(Parallel::WaitSomeNonBlocking);
      else
        nb_message_finished = m_message_list->waitMessages(Parallel::WaitSome);
      //info() << "Nb finished=" << nb_message_finished << " is_block=" << m_can_process_non_blocking;
      if (nb_message_finished == 0) {
        m_can_process_non_blocking = false;
        m_can_process_messages = true;
        _processFinishTrackingMessage();
        return;
      }
    }
  }
  m_total_time_waiting += m_timer->lastActivationTime();
  info(5) << "TimeWaiting: current=" << m_timer->lastActivationTime()
          << " total=" << m_total_time_waiting;

  // Save the messages that are currently being processed, as new messages may be added during processing
  UniqueArray<ISerializeMessage*> current_messages(m_waiting_messages);
  m_waiting_messages.clear();

  Int64UniqueArray items_to_create_id;
  Int64UniqueArray items_to_create_cells_id;
  for (Integer i = 0, is = current_messages.size(); i < is; ++i) {
    ISerializeMessage* sm = current_messages[i];
    if (sm->finished()) {
      if (!sm->isSend()) {
        _deserializeMessage(sm, items_to_create_id, items_to_create_cells_id, item_group, new_particle_local_ids);
        ++m_nb_receive_message;
      }
      delete sm;
    }
    else
      m_waiting_messages.add(sm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_sendPendingMessages()
{
  IParallelMng* pm = m_parallel_mng;

  _checkNeedReceiveMessage();

  if (!m_message_list.get())
    m_message_list = pm->createSerializeMessageListRef();

  {
    Timer::Sentry ts(m_timer);
    // Add the messages that are waiting to be processed
    Integer nb_message = m_pending_messages.size();
    for (Integer i = 0; i < nb_message; ++i) {
      m_message_list->addMessage(m_pending_messages[i]);
      m_waiting_messages.add(m_pending_messages[i]);
    }
    m_message_list->processPendingMessages();
    m_pending_messages.clear();
  }
  info(5) << "TimeSendMessages=" << m_timer->lastActivationTime()
          << " buffersize=" << m_waiting_local_ids.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_serializeMessage(ISerializeMessage* sm,
                  Int32ConstArrayView acc_ids,
                  Int64Array& items_to_send_uid,
                  Int64Array& items_to_send_cells_uid)
{
  ParticleInfoListView internal_items(m_item_family);

  ISerializer* sbuf = sm->serializer();
  sbuf->setMode(ISerializer::ModeReserve);

  //for( Integer j=0; j<nb_connected_sub_domain; ++j ){
  //ConstArrayView<Integer> acc_ids = m_ids_to_send[j];
  Integer nb_item = acc_ids.size();
  // Reserve space for the message type
  sbuf->reserveInteger(1);
  if (m_want_fast_send_particles) {
    // Reserve space for the number of particles to be sent
    sbuf->reserveInt64(1);
  }
  // Reserve space for the sender's rank
  sbuf->reserveInt32(1);
  // Reserve space for the number of uniqueId()
  sbuf->reserveInt64(1);
  // Reserve space for the uniqueId() of the particles
  sbuf->reserveSpan(eBasicDataType::Int64, nb_item);
  // Reserve space for the uniqueId() of the cells containing the particles
  sbuf->reserveSpan(eBasicDataType::Int64, nb_item);

  for (VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var;) {
    IVariable* var = *i_var;
    var->serialize(sbuf, acc_ids);
  }

  // Serialize the data for writing
  sbuf->allocateBuffer();
  sbuf->setMode(ISerializer::ModePut);

  sbuf->putInteger(MESSAGE_EXCHANGE);
  if (m_want_fast_send_particles) {
    sbuf->putInt64(m_nb_particle_finished_exchange);
    m_nb_particle_finished_exchange = 0;
  }
  sbuf->putInt32(m_rank);
  sbuf->putInt64(nb_item);
  items_to_send_uid.resize(nb_item);
  items_to_send_cells_uid.resize(nb_item);

  for (Integer z = 0; z < nb_item; ++z) {
    Particle item = internal_items[acc_ids[z]];
    items_to_send_uid[z] = item.uniqueId();
    bool has_cell = item.hasCell();
    items_to_send_cells_uid[z] = (has_cell) ? item.cell().uniqueId() : NULL_ITEM_UNIQUE_ID;
#ifdef ARCANE_DEBUG_EXCHANGE_ITEMS
#if 0
    info() << "Particle BufID=" << acc_ids[z]
           << " LID=" << item.localId()
           << " UID=" << items_to_send_uid[z]
           << " CellIUID=" << items_to_send_cells_uid[z]
           << " (owner=" << item.cell().owner() << ")";
#endif
#endif
  }
  sbuf->putSpan(items_to_send_uid);
  sbuf->putSpan(items_to_send_cells_uid);

  for (VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var;) {
    IVariable* var = *i_var;
    var->serialize(sbuf, acc_ids);
  }

  m_pending_messages.add(sm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_deserializeMessage(ISerializeMessage* message,
                    Int64Array& items_to_create_unique_id,
                    Int64Array& items_to_create_cells_id,
                    ItemGroup item_group,
                    Int32Array* new_particle_local_ids)
{

  IMesh* mesh = m_item_family->mesh();
  ISerializer* sbuf = message->serializer();

  // Set the mode to read the serialized data
  sbuf->setMode(ISerializer::ModeGet);
  sbuf->setReadMode(ISerializer::ReadReplace);
  Int32UniqueArray items_to_create_local_id;
  Int32UniqueArray cells_lid;

  Integer message_type = sbuf->getInteger();
  info(4) << "Deserialise message_type=" << (int)message_type;
  switch (message_type) {
  case MESSAGE_EXCHANGE: {
    m_need_general_receive = true;
    if (m_want_fast_send_particles) {
      Int64 nb_finished = sbuf->getInt64();
      m_nb_particle_finished_exchange += nb_finished;
    }
    Int32 orig_rank = sbuf->getInt32();
    Int64 nb_item = sbuf->getInt64();
    if (m_is_debug)
      info() << "-------------  Receive: rank=" << orig_rank << " particle nb=" << nb_item
             << " (orig_rank=" << message->destination() << ")";

    //if (nb_item!=0)
    //info() << "Receiving particules n=" << nb_item;
    items_to_create_local_id.resize(nb_item);
    items_to_create_unique_id.resize(nb_item);
    items_to_create_cells_id.resize(nb_item);
    sbuf->getSpan(items_to_create_unique_id);
    sbuf->getSpan(items_to_create_cells_id);
#ifdef ARCANE_DEBUG_EXCHANGE_ITEMS
    //info() << "Recv from SID " << sync_infos[i].subDomain() << " N=" << nb_item;
#if 0
      for( Integer z=0; z<nb_item; ++z ){
        info() << "Particle UID=" << items_to_create_unique_id[z]
               << " CellIUID=" << items_to_create_cells_id[z];
      }
#endif
#endif
    cells_lid.resize(nb_item);
    mesh->cellFamily()->itemsUniqueIdToLocalId(cells_lid, items_to_create_cells_id);

    items_to_create_local_id.resize(nb_item);
    ParticleVectorView particles_view = m_item_family->toParticleFamily()->addParticles(items_to_create_unique_id,
                                                                                        cells_lid,
                                                                                        items_to_create_local_id);
    info(5) << "Nb create=" << particles_view.size();

    // Notify the family that the modifications have been completed.
    // After this method is called, the variables can be used again.
    m_item_family->endUpdate();

    // Convert the uniqueId() values obtained into localId() values for the particles,
    // and assign the corresponding cells.
    ParticleInfoListView internal_items(m_item_family);
    //ItemInternalList internal_cells(mesh->itemsInternal(IK_Cell));
    //m_item_family->itemsUniqueIdToLocalId(items_to_create_unique_id, items_to_create_unique_id);

    for (Integer z = 0; z < nb_item; ++z) {
      Particle item = internal_items[items_to_create_local_id[z]];
      //item.setCell( internal_cells[cells_lid[z]] );
      // I am the new owner of these particles (TODO: Do not perform this here).
      item.mutableItemBase().setOwner(m_rank, m_rank);
    }
    if (!item_group.null())
      item_group.addItems(items_to_create_local_id, false);
    if (new_particle_local_ids)
      new_particle_local_ids->addRange(items_to_create_local_id);

    for (VariableCollection::Enumerator i_var(m_variables_to_exchange); ++i_var;) {
      IVariable* var = *i_var;
      var->serialize(sbuf, items_to_create_local_id);
    }
  } break;
  case MESSAGE_NB_FINISH_EXCHANGE: {
    m_need_general_receive = true;
    // Indicate that it is possible to continue receiving messages, as this message
    // does not indicate the completion of any process.
    m_can_process_messages = true;
    Int64 nb_particle = sbuf->getInt64();
    Int32 orig_rank = sbuf->getInt32();
    if (m_is_debug)
      info() << "MESSAGE_NB_FINISH_EXCHANGE nb=" << nb_particle << " (from rank=" << orig_rank << ")";
    _addFinishExchangeParticle(nb_particle);
  } break;
  case MESSAGE_FINISH_EXCHANGE_STATUS: {
    m_nb_total_particle_finish_exchange = sbuf->getInt64();
    m_exchange_finished = (m_nb_total_particle_finish_exchange == m_nb_total_particle);
    //#ifdef ARCANE_DEBUG_EXCHANGE_ITEMS
    info() << "** RECEIVING FINISH EXCHANGE " << m_exchange_finished
           << " finish=" << m_nb_total_particle_finish_exchange
           << " total=" << m_nb_total_particle;
    //#endif
    //if (m_exchange_finished)
    //warning() << "Exchange finished ! " << m_current_iteration;
  } break;
  case MESSAGE_CHANGE_BLOCKING: {
    m_need_general_receive = true;

    Integer nb_blocking_size = sbuf->getInteger();
    // It is necessary to ensure that the new blocking_size is less than the current value,
    // which may happen when multiple messages of this type are received simultaneously.
    if (nb_blocking_size < m_nb_blocking_size)
      m_nb_blocking_size = nb_blocking_size;
    info(4) << "** RECEIVING CHANGE BLOCKING"
            << " new_blocking_size=" << m_nb_blocking_size;
    // Since there may still be particles waiting to be sent, they need to be sent now.
    if (m_waiting_local_ids.size() > 0)
      _generateSendItemsMessages(Int32ConstArrayView(), Int32ConstArrayView());
  } break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
reset()
{
  if (!m_waiting_messages.empty())
    ARCANE_FATAL("reset() waiting parallel requests");
  _clearMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_processFinishTrackingMessage()
{
  // If the processor is m_master_proc:
  // - Receive the number of messages that have been completed from other processors.
  // - Calculate the total number of completed messages.
  // - Send this information to all processors.
  // If the processor is not m_master_proc:
  // - Send the value of nb_finish_tracking_particle to m_master_proc.
  // - Receive the total number of messages that have been completed from m_master_proc.
  if (m_rank == m_master_proc) {
    _addFinishExchangeParticle(m_nb_particle_finished_exchange);
  }
  else {
    // Send the number of particles to m_master_proc.
    if (m_nb_particle_finished_exchange != 0) {
      info(4) << "Send to master proc (" << m_master_proc << ") nb_finish=" << m_nb_particle_finished_exchange;
      SerializeMessage* sm = new SerializeMessage(m_rank, m_master_proc, ISerializeMessage::MT_Send);
      ISerializer* sbuf = sm->serializer();
      sbuf->setMode(ISerializer::ModeReserve);
      sbuf->reserveInteger(1);
      sbuf->reserveInt64(1);
      sbuf->reserveInt32(1);
      sbuf->allocateBuffer();
      sbuf->setMode(ISerializer::ModePut);
      sbuf->putInteger(MESSAGE_NB_FINISH_EXCHANGE);
      sbuf->putInt64(m_nb_particle_finished_exchange);
      sbuf->putInt32(m_rank);
      m_pending_messages.add(sm);
    }
  }
  m_nb_particle_finished_exchange = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_sendFinishExchangeParticle()
{
  Int32 nb_rank = m_parallel_mng->commSize();
  if (m_rank != m_master_proc || m_end_message_sended)
    return;
  m_end_message_sended = true;
  info(4) << " ** ** SEND FINISH EXCHANGE PARTICLE2";
  for (Integer i = 0; i < nb_rank; ++i) {
    if (i == m_master_proc)
      continue;
    SerializeMessage* sm = new SerializeMessage(m_rank, i, ISerializeMessage::MT_Send);
    ISerializer* sbuf = sm->serializer();
    sbuf->setMode(ISerializer::ModeReserve);
    sbuf->reserveInteger(1);
    sbuf->reserveInt64(1);
    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);
    sbuf->putInteger(MESSAGE_FINISH_EXCHANGE_STATUS);
    sbuf->putInt64(m_nb_total_particle_finish_exchange);
    m_pending_messages.add(sm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_addFinishExchangeParticle(Int64 nb_particle_finish_exchange)
{
  m_nb_total_particle_finish_exchange += nb_particle_finish_exchange;
  Int32 nb_rank = m_parallel_mng->commSize();
  Int64 nb_rank_as_int64 = nb_rank;
  //#ifdef ARCANE_DEBUG_EXCHANGE_ITEMS
  info(4) << "** RECEIVING FINISH EXCHANGE n=" << nb_particle_finish_exchange
          << " totalfinish=" << m_nb_total_particle_finish_exchange
          << " total=" << m_nb_total_particle;
  //#endif
  Int64 remaining_particle = m_nb_total_particle - m_nb_total_particle_finish_exchange;
  if (remaining_particle == 0) {
    m_exchange_finished = true;
    m_need_general_receive = false;
    info() << "** ** FINISH TRACKING NB_RECV=" << m_nb_receive_message
           << " (Date=" << platform::getCurrentDateTime() << ")";
    _sendFinishExchangeParticle();
  }
  else if (remaining_particle < (m_nb_blocking_size * nb_rank_as_int64)) {
    //Integer nb_rank = subDomain()->nbSubDomain();
    //m_nb_blocking_size /= 100;
    m_nb_blocking_size = 0;
    warning() << "** ** CHANGE BLOCKING NEW_SIZE " << m_nb_blocking_size
              << " REMAING_PARTICLE " << remaining_particle
              << " (Date=" << platform::getCurrentDateTime() << ")";

    // Since there may still be particles waiting to be sent, they need to be sent now.
    if (m_waiting_local_ids.size() > 0)
      _generateSendItemsMessages(Int32ConstArrayView(), Int32ConstArrayView());
    for (Int32 i = 0; i < nb_rank; ++i) {
      if (i == m_master_proc)
        continue;
      SerializeMessage* sm = new SerializeMessage(m_rank, i, ISerializeMessage::MT_Send);
      ISerializer* sbuf = sm->serializer();
      sbuf->setMode(ISerializer::ModeReserve);
      sbuf->reserveInteger(1);
      sbuf->reserveInteger(1);
      sbuf->allocateBuffer();
      sbuf->setMode(ISerializer::ModePut);
      sbuf->putInteger(MESSAGE_CHANGE_BLOCKING);
      sbuf->putInteger(m_nb_blocking_size);
      m_pending_messages.add(sm);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NonBlockingParticleExchanger::
_checkInitialized()
{
  if (!m_item_family)
    ARCANE_FATAL("method initialized() not called");
  _clearMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(NonBlockingParticleExchanger, IParticleExchanger,
                                           NonBlockingParticleExchanger);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
