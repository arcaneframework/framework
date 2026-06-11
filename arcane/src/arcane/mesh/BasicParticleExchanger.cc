// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicParticleExchanger.cc                                   (C) 2000-2026 */
/*                                                                           */
/* Particle Exchanger.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/BasicParticleExchanger.h"

#include "arcane/core/internal/SerializeMessage.h"

/*
 * NOTE :
 * In exchangeItems(), the array new_particle_local_ids
 * is only valid if compaction is not active for the particle family
 * (which is always the case with the current implementation).
 * For it to be valid in all cases, it would be necessary during message
 * deserialization to create all entities
 * at once, update this array \a new_particle_local_ids
 * and then update the variables.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicParticleExchanger::
BasicParticleExchanger(const ServiceBuildInfo& sbi)
: ArcaneBasicParticleExchangerObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicParticleExchanger::
~BasicParticleExchanger()
{
  // TODO: do not throw an exception in the destructor.
  if (!m_pending_messages.empty() || !m_waiting_messages.empty())
    pwarning() << String::format("Pending or waiting messages nb_pending={0} nb_waiting={1}",
                                 m_pending_messages.size(), m_waiting_messages.size());
  delete m_timer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
initialize(IItemFamily* item_family)
{
  ARCANE_CHECK_POINTER(item_family);

  m_item_family = item_family;
  m_parallel_mng = item_family->mesh()->parallelMng();
  m_rank = m_parallel_mng->commRank();
  m_timer = new Timer(m_parallel_mng->timerMng(), "BasicParticleExchanger", Timer::TimerReal);

  if (options()) {
    m_max_nb_message_without_reduce = options()->maxNbMessageWithoutReduce();
    m_support_shmem_variables = options()->supportShmemVariables();
  }

  info() << "Initialize BasicParticleExchanger family=" << item_family->name();
  info() << "-- MaxNbMessageWithoutReduce = " << m_max_nb_message_without_reduce;
  info() << "-- SupportShmemVariables = " << m_support_shmem_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_clearMessages()
{
  for (auto msg : m_accumulate_infos)
    delete msg;
  m_accumulate_infos.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
beginNewExchange(Integer i_nb_particle)
{
  _checkInitialized();
  String function_id = "BasicParticleExchanger::beginNewExchange >>> ";
  IParallelMng* pm = m_parallel_mng;

  if (options()) {
    m_debug_exchange_items_level = options()->debugExchangeItemsLevel();
    m_max_nb_message_without_reduce = options()->maxNbMessageWithoutReduce();
  }

  m_nb_loop = 0;
  m_last_nb_to_exchange = 0;
  m_exchange_finished = false;
  m_print_info = false;
  m_current_nb_reduce = 0;
  m_nb_particle_send = 0;

  //TODO: use a specific tag for this exchange.
  Int64 nb_particle = i_nb_particle;
  Int64 min_nb_particle = 0;
  Int64 max_nb_particle = 0;
  Int64 nb_total_particle = 0;
  Int32 min_rank = 0;
  Int32 max_rank = 0;
  pm->computeMinMaxSum(nb_particle, min_nb_particle, max_nb_particle,
                       nb_total_particle, min_rank, max_rank);
  //m_nb_total_particle = pm->reduce(Parallel::ReduceSum,nb_particle);
  if (m_verbose_level >= 1)
    info() << function_id << "** NB PARTICLES IN VOL total=" << nb_total_particle
           << " min=" << min_nb_particle << " max=" << max_nb_particle
           << " min_rank=" << min_rank << " max_rank=" << max_rank
           << " date=" << platform::getCurrentDateTime();

  m_nb_total_particle_finish_exchange = 0;

  // Retrieves the list of variables to transfer.
  // It is the variables that have the same family as the one passed
  // as a parameter.
  // IMPORTANT: all subdomains must have these same variables
  m_variables_to_exchange.clear();
  m_item_family->usedVariables(m_variables_to_exchange);
  m_variables_to_exchange.sortByName(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
sendItems(Integer nb_particle_finish_exchange,
          Int32ConstArrayView local_ids,
          Int32ConstArrayView sub_domains_to_send)
{
  ARCANE_UNUSED(nb_particle_finish_exchange);

  m_nb_particle_send = local_ids.size();
  {
    Timer::Sentry ts(m_timer);
    _generateSendItems(local_ids, sub_domains_to_send);
  }
  info(5) << A_FUNCINFO << "sendItems " << m_timer->lastActivationTime();

  _sendPendingMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool BasicParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send,
              ItemGroup item_group,
              IFunctor* functor)
{
  ++m_nb_loop;
  sendItems(nb_particle_finish_exchange, local_ids, sub_domains_to_send);

  if (!item_group.null())
    item_group.clear();

  return _waitMessages(0, item_group, nullptr, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool BasicParticleExchanger::
exchangeItems(Integer nb_particle_finish_exchange,
              Int32ConstArrayView local_ids,
              Int32ConstArrayView sub_domains_to_send,
              Int32Array* new_particle_local_ids,
              IFunctor* functor)
{
  ++m_nb_loop;
  sendItems(nb_particle_finish_exchange, local_ids, sub_domains_to_send);

  if (new_particle_local_ids)
    new_particle_local_ids->clear();

  return _waitMessages(0, ItemGroup(), new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_generateSendItems(Int32ConstArrayView local_ids,
                   Int32ConstArrayView sub_domains_to_send)
{
  Timer::Phase tphase(subDomain(), TP_Communication);

  IMesh* mesh = m_item_family->mesh();

  String func_name("BasicParticleExchanger::sendItems()");

  Int32UniqueArray communicating_sub_domains;
  mesh->cellFamily()->getCommunicatingSubDomains(communicating_sub_domains);

  Integer nb_connected_sub_domain = communicating_sub_domains.size();
  //Integer max_sub_domain_id = 0;
  UniqueArray<SharedArray<Int32>> ids_to_send(nb_connected_sub_domain);
  // Info for each connected subdomain
  //_clearMessages();
  m_accumulate_infos.clear();
  m_accumulate_infos.resize(nb_connected_sub_domain);
  for (Integer i = 0; i < nb_connected_sub_domain; ++i) {
    m_accumulate_infos[i] = new SerializeMessage(m_rank, communicating_sub_domains[i],
                                                 ISerializeMessage::MT_Send);
#if 0
    // Useful only for testing the timeout with blocking
    if (m_rank==0 && i==0){
      warning() << " WRONG MESSAGE";
      ISerializeMessage* sm = new SerializeMessage(m_rank,communicating_sub_domains[i],
                                                 ISerializeMessage::MT_Recv);
      m_pending_messages.add(sm);
    }
#endif
  }

  _addItemsToSend(local_ids, sub_domains_to_send, communicating_sub_domains, ids_to_send);

  if (m_debug_exchange_items_level >= 1) {
    info() << "-- Subdomain " << m_rank << ". NB to send: " << local_ids.size()
           << " NB connected subdomain: " << nb_connected_sub_domain;
    debug() << "NB connected subdomain for " << m_rank << " : " << m_accumulate_infos.size();
    for (Integer i = 0, s = m_accumulate_infos.size(); i < s; ++i) {
      debug() << "NB for the subdomain " << m_accumulate_infos[i]->destRank()
              << " " << ids_to_send[i].size();
    }
  }

  Int64UniqueArray items_to_send_uid;
  Int64UniqueArray items_to_send_cells_uid; // Only for particles;

  for (Integer j = 0; j < nb_connected_sub_domain; ++j) {
    ISerializeMessage* sm = m_accumulate_infos[j];
    // In blocking mode, always send the message because the recipient has posted
    // a receive message. Otherwise, the message only needs to be sent if it
    // contains particles.
    _serializeMessage(sm, ids_to_send[j], items_to_send_uid,
                      items_to_send_cells_uid);

    m_pending_messages.add(sm);

    // In blocking mode, we need a receive message for every send
    auto* recv_sm = new SerializeMessage(m_rank, sm->destination().value(),
                                         ISerializeMessage::MT_Recv);
    m_pending_messages.add(recv_sm);
  }

  m_accumulate_infos.clear();
  // Destroys the entities that were just sent
  m_item_family->toParticleFamily()->removeParticles(local_ids);
  m_item_family->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_addItemsToSend(Int32ConstArrayView local_ids,
                Int32ConstArrayView sub_domains_to_send,
                Int32ConstArrayView communicating_sub_domains,
                UniqueArray<SharedArray<Int32>>& ids_to_send)
{
  const Int32 debug_exchange_items_level = m_debug_exchange_items_level;
  Int32 nb_connected_sub_domain = ids_to_send.size();
  // Searches for which subdomain each element must be transferred to.
  // This search is based on the \a local_ids
  Int32 id_size = local_ids.size();
  for (Integer i = 0; i < id_size; ++i) {
    Int32 item_local_id = local_ids[i];
    Int32 sd_to_send = sub_domains_to_send[i];
    if (sd_to_send == m_rank)
      // This is an entity belonging to this subdomain
      ARCANE_FATAL("The entity with local index {0} should not be sent to its own subdomain",
                   item_local_id);
    // Searches for the index of the subdomain to which the entity belongs
    // in the list \a sync_list
    // TODO: use an indirect table (array allocated for the number of subdomains)
    Integer sd_index = nb_connected_sub_domain;
    for (Integer i_sd = 0; i_sd < nb_connected_sub_domain; ++i_sd)
      if (sd_to_send == communicating_sub_domains[i_sd]) {
        sd_index = i_sd;
        break;
      }
    if (sd_index == nb_connected_sub_domain)
      ARCANE_FATAL("Internal: bad rank index");
    ids_to_send[sd_index].add(item_local_id);
    if (debug_exchange_items_level >= 1)
      pinfo() << "ADD ITEM TO SEND lid=" << item_local_id << " sd_index=" << sd_index
              << " sd=" << communicating_sub_domains[sd_index] << " n=" << ids_to_send[sd_index].size()
              << " begin=" << ids_to_send[sd_index].data();
  }
  if (debug_exchange_items_level >= 1)
    for (Integer i = 0; i < nb_connected_sub_domain; ++i)
      pinfo() << "SEND INFO sd_index=" << i
              << " sd=" << communicating_sub_domains[i] << " n=" << ids_to_send[i].size()
              << " begin=" << ids_to_send[i].data();

  if ((debug_exchange_items_level >= 1 && m_print_info) || debug_exchange_items_level >= 2) {
    IParallelMng* pm = m_parallel_mng;
    Int32 rank = pm->commRank();
    ParticleInfoListView items(m_item_family);
    Integer nb_print = math::min(5, id_size);
    for (Integer i = 0; i < nb_print; ++i) {
      Particle part(items[local_ids[i]]);
      pinfo() << " RANK=" << rank << " LID=" << local_ids[i] << " SD=" << sub_domains_to_send[i]
              << " uid=" << part.uniqueId() << " cell=" << ItemPrinter(part.cell());
    }
    pm->barrier();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool BasicParticleExchanger::
waitMessages(Integer nb_pending_particle, Int32Array* new_particle_local_ids, IFunctor* functor)
{
  return _waitMessages(nb_pending_particle, ItemGroup(), new_particle_local_ids, functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool BasicParticleExchanger::
_waitMessages(Integer nb_pending_particle, ItemGroup item_group,
              Int32Array* new_particle_local_ids, IFunctor* functor)
{
  String func_name = "BasicParticleExchanger::waitMessages";
  _waitMessages(item_group, new_particle_local_ids, functor);

  bool do_reduce = m_current_nb_reduce > m_last_nb_reduce;
  if (m_max_nb_message_without_reduce != (-1))
    do_reduce |= m_current_nb_reduce > m_max_nb_message_without_reduce;

  if (do_reduce) {
    Int64 nb_to_exchange = 0;
    IParallelMng* pm = m_parallel_mng;
    {
      Timer::Sentry ts(m_timer);
      Int64 current_exchange = m_nb_particle_send + nb_pending_particle;
      nb_to_exchange = pm->reduce(Parallel::ReduceSum, current_exchange);
    }
    info(4) << func_name << "TimeReduce=" << m_timer->lastActivationTime()
            << " nbtoexchange=" << nb_to_exchange;
    m_exchange_finished = (nb_to_exchange == 0);
    if (nb_to_exchange > 0 && m_last_nb_to_exchange == nb_to_exchange && m_nb_loop > 300) {
      m_print_info = true;
    }
    m_last_nb_to_exchange = nb_to_exchange;
  }

  ++m_current_nb_reduce;
  debug() << func_name << " ** RETURN EXCHANGE m_exchange_finished: " << m_exchange_finished;
  if (m_exchange_finished) {
    m_last_nb_reduce = m_current_nb_reduce - 4;
    if (m_verbose_level >= 1)
      info() << func_name << " exchange finished "
             << " n=" << m_current_nb_reduce
             << " date=" << platform::getCurrentDateTime();
  }
  return m_exchange_finished;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_waitMessages(ItemGroup item_group, Int32Array* new_particle_local_ids, IFunctor* functor)
{
  if (functor) {
    {
      Timer::Sentry ts(m_timer);
      functor->executeFunctor();
    }
    m_total_time_functor += m_timer->lastActivationTime();
    if (m_debug_exchange_items_level >= 1)
      info() << "TimeFunctor: current=" << m_timer->lastActivationTime()
             << " total=" << m_total_time_functor;
  }

  {
    Timer::Sentry ts(m_timer);
    m_message_list->waitMessages(Parallel::WaitAll);
  }
  m_total_time_waiting += m_timer->lastActivationTime();
  if (m_debug_exchange_items_level >= 1)
    info() << "TimeWaiting: current=" << m_timer->lastActivationTime()
           << " total=" << m_total_time_waiting;

  // Saves the currently processed communications because the processing
  // might add new ones
  UniqueArray<ISerializeMessage*> current_messages(m_waiting_messages);
  m_waiting_messages.clear();

  Int32 nb_end_update = 0;
  Int32 max_nb_messages = 0;
  if (m_support_shmem_variables) {
    max_nb_messages = m_parallel_mng->reduce(MessagePassing::ReduceMax, current_messages.size());
  }

  Int64UniqueArray items_to_create_unique_id;
  Int64UniqueArray items_to_create_cells_unique_id;
  Int32UniqueArray items_to_create_local_id;
  Int32UniqueArray items_to_create_cells_local_id;
  for (ISerializeMessage* sm : current_messages) {
    if (!sm->isSend()) {
      _deserializeMessage(sm, items_to_create_unique_id, items_to_create_cells_unique_id,
                          items_to_create_local_id, items_to_create_cells_local_id,
                          item_group, new_particle_local_ids);
      nb_end_update++;
    }
    delete sm;
  }

  if (m_support_shmem_variables) {
    for (; nb_end_update < max_nb_messages; ++nb_end_update) {
      m_item_family->endUpdate();
    }
  }

  if (!m_waiting_messages.empty())
    ARCANE_FATAL("Pending messages n={0}", m_waiting_messages.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_sendPendingMessages()
{
  IParallelMng* pm = m_parallel_mng;

  if (!m_message_list.get())
    m_message_list = pm->createSerializeMessageListRef();

  {
    Timer::Sentry ts(m_timer);
    // Adds the messages waiting for processing
    Integer nb_message = m_pending_messages.size();
    for (Integer i = 0; i < nb_message; ++i) {
      m_message_list->addMessage(m_pending_messages[i]);
      m_waiting_messages.add(m_pending_messages[i]);
    }
    m_message_list->processPendingMessages();
    m_pending_messages.clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
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
  // Reserves for the message type
  //sbuf->reserveInteger(1);
  // Reserves for the message ID
  sbuf->reserveInt64(1);
  // Reserves for the number of uniqueId()
  sbuf->reserveInt64(1);
  sbuf->reserveSpan(eBasicDataType::Int64, nb_item);
  // Reserves for the uniqueIds() of the cells where the particles are located
  //sbuf->reserve(DT_Size,1);
  sbuf->reserveSpan(eBasicDataType::Int64, nb_item);

  for (VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var;) {
    IVariable* var = *i_var;
    var->serialize(sbuf, acc_ids);
  }

  // Serializes the data for writing
  sbuf->allocateBuffer();

  if (m_debug_exchange_items_level >= 1)
    info() << "BSE_SerializeMessage nb_item=" << nb_item
           << " id=" << m_serialize_id
           << " dest=" << sm->destination();

  sbuf->setMode(ISerializer::ModePut);

  sbuf->putInt64(m_serialize_id);
  ++m_serialize_id;

  sbuf->putInt64(nb_item);
  items_to_send_uid.resize(nb_item);
  items_to_send_cells_uid.resize(nb_item);
  for (Integer z = 0; z < nb_item; ++z) {
    Particle item = internal_items[acc_ids[z]];
    items_to_send_uid[z] = item.uniqueId();
    bool has_cell = item.hasCell();
    items_to_send_cells_uid[z] = (has_cell) ? item.cell().uniqueId() : NULL_ITEM_UNIQUE_ID;
    if (m_debug_exchange_items_level >= 2) {
      info() << "Particle BufID=" << acc_ids[z]
             << " LID=" << item.localId()
             << " UID=" << items_to_send_uid[z]
             << " CellIUID=" << items_to_send_cells_uid[z]
             << " (owner=" << item.cell().owner() << ")";
    }
  }
  sbuf->putSpan(items_to_send_uid);
  sbuf->putSpan(items_to_send_cells_uid);

  for (VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var;) {
    IVariable* var = *i_var;
    var->serialize(sbuf, acc_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_deserializeMessage(ISerializeMessage* message,
                    Int64Array& items_to_create_unique_id,
                    Int64Array& items_to_create_cells_unique_id,
                    Int32Array& items_to_create_local_id,
                    Int32Array& items_to_create_cells_local_id,
                    ItemGroup item_group,
                    Int32Array* new_particle_local_ids)
{

  IMesh* mesh = m_item_family->mesh();
  ISerializer* sbuf = message->serializer();
  IItemFamily* cell_family = mesh->cellFamily();

  // Indicates that we want to serialize the data for reading
  sbuf->setMode(ISerializer::ModeGet);
  sbuf->setReadMode(ISerializer::ReadReplace);

  {
    Int64 serialize_id = sbuf->getInt64();
    Int64 nb_item = sbuf->getInt64();
    if (m_debug_exchange_items_level >= 1)
      info() << "BSE_DeserializeMessage id=" << serialize_id << " nb=" << nb_item
             << " orig=" << message->destination();

    items_to_create_local_id.resize(nb_item);
    items_to_create_unique_id.resize(nb_item);
    items_to_create_cells_unique_id.resize(nb_item);
    items_to_create_cells_local_id.resize(nb_item);
    sbuf->getSpan(items_to_create_unique_id);
    sbuf->getSpan(items_to_create_cells_unique_id);
    if (m_debug_exchange_items_level >= 2) {
      //info() << "Recv from SID " << sync_infos[i].subDomain() << " N=" << nb_item;
      for (Integer z = 0; z < nb_item; ++z) {
        info() << "Particle UID=" << items_to_create_unique_id[z]
               << " CellIUID=" << items_to_create_cells_unique_id[z];
      }
    }

    items_to_create_cells_local_id.resize(nb_item);
    cell_family->itemsUniqueIdToLocalId(items_to_create_cells_local_id, items_to_create_cells_unique_id);

    m_item_family->toParticleFamily()->addParticles(items_to_create_unique_id,
                                                    items_to_create_cells_local_id,
                                                    items_to_create_local_id);

    // Notifies the family that we have finished our modifications.
    // After calling this method, the variables are usable again
    m_item_family->endUpdate();

    // Converts the retrieved uniqueId() to localId() and for particles
    // populates the corresponding cell
    ParticleInfoListView internal_items(m_item_family);

    for (Integer z = 0; z < nb_item; ++z) {
      Particle item = internal_items[items_to_create_local_id[z]];
      //item.setCell( internal_cells[items_to_create_cells_local_id[z]] );
      // I am the new owner (TODO: do not do this here)
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
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
reset()
{
  if (!m_waiting_messages.empty())
    ARCANE_FATAL("waiting parallel requests");
  _clearMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
addNewParticles(Integer nb_particle)
{
  ARCANE_UNUSED(nb_particle);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchanger::
_checkInitialized()
{
  if (!m_item_family)
    ARCANE_FATAL("method initialized() not called");
  _clearMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_BASICPARTICLEEXCHANGER(BasicParticleExchanger, BasicParticleExchanger);

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(BasicParticleExchanger, IParticleExchanger,
                                   BasicParticleExchanger);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
