// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GetVariablesValuesParallelOperation.cc                      (C) 2000-2025 */
/*                                                                           */
/* Opérations pour accéder aux valeurs de variables d'un autre sous-domaine. */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/Timer.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ISerializeMessage.h"

#include "arcane/impl/GetVariablesValuesParallelOperation.h"

#include "arccore/message_passing/ISerializeMessageList.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GetVariablesValuesParallelOperation::
GetVariablesValuesParallelOperation(IParallelMng* pm)
: m_parallel_mng(pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* GetVariablesValuesParallelOperation::
parallelMng()
{
  return m_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GetVariablesValuesParallelOperation::
getVariableValues(VariableItemReal& variable,
                  ConstArrayView<Int64> unique_ids,
                  ConstArrayView<Int32> sub_domain_ids,
                  ArrayView<Real> values)
{
  IParallelMng* pm = m_parallel_mng;
  Timer::Phase tphase(pm->timeStats(), TP_Communication);

  if (!pm->isParallel()) {
    _getVariableValuesSequential(variable, unique_ids, values);
    return;
  }

  ItemGroup group = variable.itemGroup();
  IItemFamily* item_family = variable.variable()->itemFamily();

  if (group.null())
    ARCANE_FATAL("The variable '{0}' is not defined on a group.", variable.name());

  Int32 nb_item = unique_ids.size();
  if (nb_item != values.size())
    ARCANE_FATAL("The arrays 'unique_ids' and 'values' don't have the same "
                 "number of elements (respectively {0} and {1}).",
                 nb_item, values.size());

  if (nb_item != sub_domain_ids.size())
    ARCANE_FATAL("The arrays 'unique_ids' and 'sub_domains_ids' don't have the same "
                 "number of elements (respectively {0} and {1}).",
                 nb_item, sub_domain_ids.size());

  using SubDomainUniqueIdMap = std::map<Int32, Helper>;
  SubDomainUniqueIdMap sub_domain_list;

  for (Integer i = 0; i < nb_item; ++i) {
    Int32 sd = sub_domain_ids[i];
    if (sd == NULL_SUB_DOMAIN_ID)
      ARCANE_FATAL("Null SubDomainId for index {0}", i);
    //TODO ne pas ajouter les éléments de son propre sous-domaine à la liste
    Helper& h = sub_domain_list[sd];
    h.m_unique_ids.add(unique_ids[i]);
    h.m_indexes.add(i);
  }

  UniqueArray<Int32> sub_domain_nb_to_send;
  Int32 my_rank = pm->commRank();
  for (auto& [sd, helper] : sub_domain_list) {
    Integer n = helper.m_unique_ids.size();
    sub_domain_nb_to_send.add(my_rank);
    sub_domain_nb_to_send.add(sd);
    sub_domain_nb_to_send.add(n);
  }

  UniqueArray<Int32> total_sub_domain_nb_to_send;
  pm->allGatherVariable(sub_domain_nb_to_send, total_sub_domain_nb_to_send);
  UniqueArray<Ref<ISerializeMessage>> messages;
  Ref<ISerializeMessageList> message_list(pm->createSerializeMessageListRef());
  for (Integer i = 0, is = total_sub_domain_nb_to_send.size(); i < is; i += 3) {
    Int32 rank_send = total_sub_domain_nb_to_send[i];
    Int32 rank_recv = total_sub_domain_nb_to_send[i + 1];
    //Integer nb_exchange = total_sub_domain_nb_to_send[i+2];
    //trace->info() << " SEND=" << rank_send
    //<< " RECV= " << rank_recv
    //<< " N= " << nb_exchange;
    if (rank_send == rank_recv)
      continue;
    Ref<ISerializeMessage> sm;
    if (rank_recv == my_rank) {
      //trace->info() << " ADD RECV MESSAGE recv=" << rank_recv << " send=" << rank_send;
      sm = message_list->createAndAddMessage(MessageRank(rank_send), ePointToPointMessageType::MsgReceive);
    }
    else if (rank_send == my_rank) {
      //trace->info() << " ADD SEND MESSAGE recv=" << rank_recv << " send=" << rank_send;
      sm = message_list->createAndAddMessage(MessageRank(rank_recv), ePointToPointMessageType::MsgSend);
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeReserve);
      auto xiter = sub_domain_list.find(rank_recv);
      if (xiter == sub_domain_list.end())
        ARCANE_FATAL("Can not find rank '{0}'", rank_recv);
      Span<const Int64> z_unique_ids = xiter->second.m_unique_ids;
      Int64 nb = z_unique_ids.size();
      s->reserveInt64(1); // Pour la taille
      s->reserveSpan(eBasicDataType::Int64, nb); // Pour le tableau
      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);
      s->putInt64(nb);
      s->putSpan(z_unique_ids);
    }
    if (sm.get())
      messages.add(sm);
  }

  message_list->waitMessages(eWaitType::WaitAll);

  UniqueArray<Int64> tmp_unique_ids;
  UniqueArray<Int32> tmp_local_ids;
  UniqueArray<Real> tmp_values;

  UniqueArray<Ref<ISerializeMessage>> values_messages;
  ItemInfoListView items_internal(item_family);
  for (Ref<ISerializeMessage> sm : messages) {
    Ref<ISerializeMessage> new_sm;
    if (sm->isSend()) {
      // Pour recevoir les valeurs
      //trace->info() << " ADD RECV2 MESSAGE recv=" << my_rank << " send=" << sm->destSubDomain();
      new_sm = message_list->createAndAddMessage(MessageRank(sm->destination().value()), ePointToPointMessageType::MsgReceive);
    }
    else {
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb = s->getInt64();
      tmp_unique_ids.resize(nb);
      tmp_local_ids.resize(nb);
      tmp_values.resize(nb);
      s->getSpan(tmp_unique_ids);
      item_family->itemsUniqueIdToLocalId(tmp_local_ids, tmp_unique_ids);
      for (Integer z = 0; z < nb; ++z) {
        Item item = items_internal[tmp_local_ids[z]];
        tmp_values[z] = variable[item];
      }

      //trace->info() << " ADD SEND2 MESSAGE recv=" << my_rank << " send=" << sm->destSubDomain();
      new_sm = message_list->createAndAddMessage(MessageRank(sm->destination().value()), ePointToPointMessageType::MsgSend);
      ISerializer* s2 = new_sm->serializer();
      s2->setMode(ISerializer::ModeReserve);
      s2->reserveInt64(1);
      s2->reserveSpan(eBasicDataType::Real, nb);
      s2->allocateBuffer();
      s2->setMode(ISerializer::ModePut);
      s2->putInt64(nb);
      s2->putSpan(tmp_values);
    }
    values_messages.add(new_sm);
  }

  // Supprime les messages qui ne sont plus utilisés
  messages.clear();

  message_list->waitMessages(eWaitType::WaitAll);

  for (Ref<ISerializeMessage> sm : values_messages) {
    if (sm->isSend()) {
    }
    else {
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb = s->getInt64();
      tmp_values.resize(nb);
      Int32 sender = sm->destination().value();
      s->getSpan(tmp_values);
      //trace->info() << " GET VALUES from=" << sm->destSubDomain() << " n=" << nb;
      Span<const Int32> indexes = sub_domain_list[sender].m_indexes;
      for (Int64 z = 0; z < nb; ++z)
        values[indexes[z]] = tmp_values[z];
    }
  }

  // Enfin, traite ses propres éléments
  // TODO: FAIRE CE TRAITEMENT EN ATTENDANT LES MESSAGES
  {
    Helper h(sub_domain_list[my_rank]);
    Span<const Int32> indexes(h.m_indexes.constSpan());
    Int64 nb = h.m_unique_ids.largeSize();
    tmp_local_ids.resize(nb);
    item_family->itemsUniqueIdToLocalId(tmp_local_ids, h.m_unique_ids);
    for (Int64 z = 0; z < nb; ++z) {
      Item item = items_internal[tmp_local_ids[z]];
      values[indexes[z]] = variable[item];
    }
  }

  // Supprime les messages qui ne sont plus utilisés
  values_messages.clear();
  //_deleteMessages(values_messages);

#if 0
  {
    // Pour faire une petite vérification
    Integer nb_values = values.size();
    RealUniqueArray ref_values(nb_values);
    ref_values.fill(0.0);
    getVariableValues(variable,unique_ids,ref_values);
    bool has_error = false;
    for( Integer i=0; i<nb_values; ++i ){
      if (!math::isEqual(ref_values[i],values[i])){
        trace->pinfo() << " Incorrect values ref=" << ref_values[i] << " v=" << values[i];
        has_error = true;
      }
    }
    if (has_error)
      trace->fatal() << func_id << " incorrect values";
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GetVariablesValuesParallelOperation::
getVariableValues(VariableItemReal& variable, Int64ConstArrayView unique_ids,
                  RealArrayView values)
{
  IParallelMng* pm = m_parallel_mng;
  Timer::Phase tphase(pm->timeStats(), TP_Communication);

  if (!pm->isParallel()) {
    _getVariableValuesSequential(variable, unique_ids, values);
    return;
  }

  ItemGroup group = variable.itemGroup();
  ITraceMng* trace = pm->traceMng();
  if (group.null())
    ARCANE_FATAL("The variable '{0}' is not defined on a group.", variable.name());

  Int32 size = unique_ids.size();
  if (size != values.size())
    ARCANE_FATAL("The arrays 'unique_ids' and 'values' don't have the same "
                 "number of elements (respectively {0} and {1}).",
                 size, values.size());

  Int32 nb_proc = pm->commSize();
  Integer nb_phase = 0;
  while (nb_proc != 0 && nb_phase < 32) {
    nb_proc /= 2;
    ++nb_phase;
  }
  if (nb_phase < 3)
    nb_phase = 1;
  trace->info() << " NB PHASE=" << nb_phase;
  nb_phase = 1;
  if (nb_phase == 1) {
    _getVariableValues(variable, unique_ids, values);
  }
  else {
    Integer nb_done = 0;
    for (Integer i = 0; i < nb_phase; ++i) {
      Integer first = (i * size) / nb_phase;
      Integer last = ((i + 1) * size) / nb_phase;
      if ((i + 1) == nb_phase)
        last = size;
      Integer n = last - first;
      nb_done += n;
      trace->debug() << "GetVariableValue: first=" << first << " last=" << last << " n=" << n
                     << " size=" << size;
      RealArrayView local_values(n, values.data() + first);
      Int64ConstArrayView local_unique_ids(n, unique_ids.data() + first);
      _getVariableValues(variable, local_unique_ids, local_values);
    }
    if (nb_done != size) {
      trace->fatal() << "MpiParallelMng::getVariableValue() Internal error in size: "
                     << " size=" << size << " done=" << nb_done;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> void GetVariablesValuesParallelOperation::
_getVariableValues(ItemVariableScalarRefT<Type>& variable,
                   Int64ConstArrayView unique_ids,
                   ArrayView<Type> values)
{
  IParallelMng* pm = m_parallel_mng;
  ItemGroup group = variable.itemGroup();
  ITraceMng* msg = pm->traceMng();
  IItemFamily* item_family = group.itemFamily();

  // Pour éviter un bug MPI sur certaines machines,
  // si la liste est vide, on crée une liste temporaire
  UniqueArray<Int64> dummy_unique_ids;
  UniqueArray<Real> dummy_values;
  if (unique_ids.empty()) {
    dummy_unique_ids.resize(1);
    dummy_values.resize(1);
    dummy_unique_ids[0] = NULL_ITEM_ID;
    unique_ids = dummy_unique_ids.view();
    values = dummy_values.view();
  }

  // Principe de fonctionnement.
  // Chaque sous-domaine récupère la totalité des unique_ids dont on veut
  // les valeurs (allGatherVariable).
  // On alloue ensuite un tableau dimensionné à ce nombre de uniqueId() qui
  // contiendra les valeurs des entités (tableau all_value).
  // Chaque sous-domaine remplit ce tableau comme suit :
  // * si l'entité lui appartient, remplit avec la valeur de la variable
  // * sinon, remplit avec la valeur minimale possible suivant \a Type.
  // Le processeur 0 effectue ensuite une réduction Max de ce tableau,
  // qui contiendra alors la bonne valeur pour chacun de ses éléments.
  // Il ne reste plus alors qu'à faire un 'scatter' symétrique du
  // premier 'gather'.

  Int64UniqueArray all_unique_ids;
  pm->allGatherVariable(unique_ids, all_unique_ids);
  Integer all_size = all_unique_ids.size();
  Int32UniqueArray all_local_ids(all_size);
  item_family->itemsUniqueIdToLocalId(all_local_ids, all_unique_ids, false);

  ConstArrayView<Type> variable_a(variable.asArray());
  UniqueArray<Type> all_values(all_size);

  msg->debug() << "MpiParallelMng::_getVariableValues(): size=" << all_size
               << " values_size=" << sizeof(Type) * all_size;

  // Remplit le tableau des valeurs avec la valeur maximale possible
  // pour le type. Il suffit ensuite de faire un ReduceMin
  Type max_value = std::numeric_limits<Type>::max();
  ItemInfoListView internal_items(item_family);

  for (Integer i = 0; i < all_size; ++i) {
    Integer lid = all_local_ids[i];
    if (lid == NULL_ITEM_ID)
      all_values[i] = max_value;
    else {
      all_values[i] = (internal_items[lid].isOwn()) ? variable_a[lid] : max_value;
    }
  }

  pm->reduce(Parallel::ReduceMin, all_values);

  // Scinde le tableau sur les autres processeurs
  pm->scatterVariable(all_values, values, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> void GetVariablesValuesParallelOperation::
_getVariableValuesSequential(ItemVariableScalarRefT<Type>& variable,
                             Int64ConstArrayView unique_ids,
                             ArrayView<Type> values)
{
  ItemGroup group = variable.itemGroup();
  if (group.null())
    ARCANE_FATAL("The variable '{0}' is not defined on a group.", variable.name());

  IItemFamily* family = group.itemFamily();
  Int32 size = unique_ids.size();
  if (size != values.size())
    ARCANE_FATAL("The arrays 'unique_ids' and 'values' don't have the same "
                 "number of elements (respectively {0} and {1}).",
                 size, values.size());

  //TODO: faire par morceaux.
  UniqueArray<Int32> local_ids(size);
  family->itemsUniqueIdToLocalId(local_ids, unique_ids);
  ConstArrayView<Type> variable_a(variable.asArray());
  for (Integer i = 0; i < size; ++i)
    values[i] = variable_a[local_ids[i]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
