// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsExchangeInfo2.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Exchange of entities and their variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemsExchangeInfo2.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/ItemFamilySerializeArgs.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

#include "arcane/mesh/ItemGroupsSerializer2.h"
#include "arcane/mesh/TiedInterfaceExchanger.h"
#include "arcane/mesh/ItemFamilyVariableSerializer.h"

// TODO: to be removed
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

#include "arcane/IVariableAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemFamilyExchange;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
namespace
{
  const Integer GROUPS_MAGIC_NUMBER = 0x3a9e4325;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsExchangeInfo2::
ItemsExchangeInfo2(IItemFamily* item_family)
: TraceAccessor(item_family->traceMng())
, m_item_family(item_family)
, m_groups_serializers()
, m_exchanger(ParallelMngUtils::createExchangerRef(item_family->parallelMng()))
, m_family_serializer(nullptr)
{
  m_family_serializer = item_family->policyMng()->createSerializer();

  // Positionne infos pour l'affichage listing
  m_exchanger->setName(item_family->name());

  // Celui ci doit toujours être le premier de la phase de sérialisation des variables.
  addSerializeStep(new ItemFamilyVariableSerializer(item_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsExchangeInfo2::
~ItemsExchangeInfo2()
{
  for (IItemFamilySerializeStep* step : m_serialize_steps)
    delete step;
  delete m_family_serializer;
  for (Integer i = 0; i < m_groups_serializers.size(); ++i)
    delete m_groups_serializers[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void ItemsExchangeInfo2::
_addItemToSend(Int32 sub_domain_id, Item item)
{
  if (m_send_local_ids[sub_domain_id].empty())
    // If it is the first element, add the sub-domain to the list of
    // communicating sub-domains
    m_exchanger->addSender(sub_domain_id);
  m_send_local_ids[sub_domain_id].add(item.localId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemsExchangeInfo2::
computeExchangeInfos()
{
  // Determines the list of variables to exchange
  // It also includes variables from child families
  {
    m_families_to_exchange.add(itemFamily()); // The current family
    IItemFamilyCollection child_families = itemFamily()->childFamilies();
    for (IItemFamily* child_family : child_families)
      m_families_to_exchange.add(child_family);

    for (IItemFamily* current_family : m_families_to_exchange) {
      // If the family does not have a uniqueId table, it must not
      // transfer the groups because it is not possible to convert
      // uniqueIds to localIds and the serializer needs it.
      // TODO: remove this requirement in the serializer
      if (current_family->hasUniqueIdMap())
        m_groups_serializers.add(new ItemGroupsSerializer2(current_family, m_exchanger.get()));
    }
  }

  for (IItemFamilySerializeStep* step : m_serialize_steps) {
    step->initialize();
  }

  bool r = m_exchanger->initializeCommunicationsMessages();
  m_receive_local_ids.resize(m_exchanger->nbReceiver());
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
setExchangeItems(ConstArrayView<std::set<Int32>> items_to_send)
{
  Int32 nb_part = m_item_family->mesh()->meshPartInfo().nbPart();
  m_send_local_ids.resize(nb_part);

  for (Integer i = 0, is = items_to_send.size(); i < is; ++i) {
    Int64 n = items_to_send[i].size();
    if (n != 0) {
      m_exchanger->addSender(i);
      //ItemInternalList items = m_item_family->itemsInternal();
      std::set<Int32>::const_iterator iids = items_to_send[i].begin();
      std::set<Int32>::const_iterator ids_end = items_to_send[i].end();
      //info() << "SEND kind=" << itemKindName(itemKind()) << " dest=" << i << " nb=" << n;
      for (; iids != ids_end; ++iids) {
        m_send_local_ids[i].add(*iids);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Determines the list of entities to exchange.

 * \warning This method should only be used for particle families.

 This operation uses the itemsOwner() variable and the owner() field
 of each entity to determine who each entity must
 be sent to. Therefore, this operation must be called
 before DynamicMesh::_setOwnerFromVariable() is called.

 TODO: move this method elsewhere.
*/
void ItemsExchangeInfo2::
computeExchangeItems()
{
  if (m_item_family->itemKind() != IK_Particle)
    ARCANE_FATAL("This call is only valid for ParticleFamily. family={0}",
                 itemFamily()->name());

  Int32 nb_part = m_item_family->mesh()->meshPartInfo().nbPart();
  // Contains for each sub-domain the list of entities to send
  m_send_local_ids.resize(nb_part);
  // List of sub-domains with which I must communicate.
  VariableItemInt32& items_owner(itemFamily()->itemsNewOwner());

  // To determine the list of entities to send, it is sufficient to compare
  // the owner() field of the entity containing the owner of the current entity
  // with the itemsNewOwner() variable which contains the new owner.
  // If these two values are different, the entity must be sent.
  // WARNING: above all, do not use ownItems(), because these are calculated on the fly
  // (lazy evaluation)
  ENUMERATE_ITEM (i_item, itemFamily()->allItems()) {
    Item item = *i_item;
    Int32 new_owner = items_owner[item];
    Int32 current_owner = item.owner();
    if (item.isOwn() && new_owner != current_owner) {
      _addItemToSend(new_owner, item);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
prepareToSend()
{
  info(4) << "ItemsExchangeInfo2::prepareToSend() for " << itemFamily()->name();
  info(4) << "Number of groups to serialize: " << m_groups_serializers.size();

  // Preparation of group serializers
  for (Integer i_serializer = 0; i_serializer < m_groups_serializers.size(); ++i_serializer) {
    ItemGroupsSerializer2* groups_serializer = m_groups_serializers[i_serializer];
    if (groups_serializer->itemFamily() == itemFamily()) {
      // This is the original family, so no transformation
      groups_serializer->prepareData(m_send_local_ids);
    }
    else {
      // This is a sub-family
      UniqueArray<SharedArray<Int32>> subitems_to_send(m_send_local_ids.size());
      for (Integer i_dest = 0; i_dest < m_send_local_ids.size(); ++i_dest) {
        ItemVector subitems = MeshToMeshTransposer::transpose(itemFamily(), groups_serializer->itemFamily(), itemFamily()->view(m_send_local_ids[i_dest]));
        Int32Array& current_subitem_lids = subitems_to_send[i_dest];
        ENUMERATE_ITEM (iitem, subitems)
          if (iitem.localId() != NULL_ITEM_LOCAL_ID)
            current_subitem_lids.add(iitem.localId());
      }
      groups_serializer->prepareData(subitems_to_send);
    }
  }

  // Generates info for each processor to which entities will be sent
  ItemInfoListView items_internal(itemFamily());
  IItemFamilyCollection child_families = itemFamily()->childFamilies();

  const Integer nb_send = m_exchanger->nbSender();
  {
    auto action = IItemFamilySerializeStep::eAction::AC_BeginPrepareSend;
    for (IItemFamilySerializeStep* step : m_serialize_steps)
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action, nb_send));
  }

  for (Integer i = 0; i < nb_send; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToSend(i);
    Int32 dest_sub_domain = comm->destination().value();
    // List of localId() of entities to send
    Int32ConstArrayView dest_items_local_id = m_send_local_ids[dest_sub_domain];
    info(5) << "Processing message to " << dest_sub_domain
            << " for family " << itemFamily()->fullName();

    ISerializer* sbuf = comm->serializer();

    ItemFamilySerializeArgs serialize_args(sbuf, dest_sub_domain, dest_items_local_id, i);

    // Reserves memory for serialization
    sbuf->setMode(ISerializer::ModeReserve);

    // Reserves for the items and uids of the sub-items
    m_family_serializer->serializeItems(sbuf, dest_items_local_id);
    m_family_serializer->serializeItemRelations(sbuf, dest_items_local_id);

    // Reserves for the uids of the sub-items (duplicate calculation of MeshToMeshTransposer::transpose with the put)
    for (IItemFamily* child_family : child_families) {
      ItemVectorView dest_items(items_internal, dest_items_local_id);
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(itemFamily(), child_family, dest_items);
      Integer sub_dest_item_count = 0;
      ENUMERATE_ITEM (iitem, sub_dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID)
          ++sub_dest_item_count;
      }
      sbuf->reserveInt64(1);
      sbuf->reserveSpan(eBasicDataType::Int64, sub_dest_item_count);
    }

    _applySerializeStep(IItemFamilySerializeStep::PH_Item, serialize_args);

    sbuf->reserveInteger(1); // For magic number for group serialization

    // Reserves for the groups
    for (Integer i_serializer = 0; i_serializer < m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->serialize(serialize_args);

    _applySerializeStep(IItemFamilySerializeStep::PH_Group, serialize_args);

    // The following objects are deserialized in readVariables()

    // Reserves for variable serialization
    _applySerializeStep(IItemFamilySerializeStep::PH_Variable, serialize_args);

    sbuf->allocateBuffer();

    // Serializes the info
    sbuf->setMode(ISerializer::ModePut);

    m_family_serializer->serializeItems(sbuf, dest_items_local_id);
    m_family_serializer->serializeItemRelations(sbuf, dest_items_local_id);

    // Serialization of sub-item uids (duplicate calculation of MeshToMeshTransposer::transpose with reserves)
    for (IItemFamily* child_family : child_families) {
      ItemVectorView dest_items(items_internal, dest_items_local_id);
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(itemFamily(), child_family, dest_items);
      Int64UniqueArray sub_dest_uids;
      sub_dest_uids.reserve(sub_dest_items.size());
      ENUMERATE_ITEM (iitem, sub_dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID)
          sub_dest_uids.add(iitem->uniqueId());
      }
      sbuf->putInt64(sub_dest_uids.size());
      sbuf->putSpan(sub_dest_uids);
    }

    _applySerializeStep(IItemFamilySerializeStep::PH_Item, serialize_args);

    sbuf->put(GROUPS_MAGIC_NUMBER);

    // Serializes the list of groups
    for (Integer i_serializer = 0; i_serializer < m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->serialize(serialize_args);

    _applySerializeStep(IItemFamilySerializeStep::PH_Group, serialize_args);

    // Serializes the info for variables
    _applySerializeStep(IItemFamilySerializeStep::PH_Variable, serialize_args);
  }

  {
    auto action = IItemFamilySerializeStep::eAction::AC_EndPrepareSend;
    for (IItemFamilySerializeStep* step : m_serialize_steps)
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action, nb_send));
  }
}

void ItemsExchangeInfo2::
releaseBuffer()
{
  for (Integer i = 0, is = m_exchanger->senderRanks().size(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToSend(i);

    ISerializer* isbuf = comm->serializer();
    SerializeBuffer* sbuf = dynamic_cast<SerializeBuffer*>(isbuf);

    if (sbuf)
      sbuf->releaseBuffer();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readAndAllocItems()
{
  info(4) << "ItemsExchangeInfo2::readAndAllocItems() " << itemFamily()->name();

  // The organization of the loops and the switch is not identical here to prepareToSend; for readability, they should be made similar.

  // Retrieves the info of the meshes of each receiver and creates the entities.
  for (Integer i = 0, is = m_exchanger->nbReceiver(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    info(5) << "Processing item message from " << comm->destination()
            << " for family " << itemFamily()->fullName();
    m_family_serializer->deserializeItems(sbuf, &m_receive_local_ids[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readAndAllocSubMeshItems()
{
  IItemFamilyCollection child_families = itemFamily()->childFamilies();
  for (Integer i = 0, is = m_exchanger->nbReceiver(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();

    for (IItemFamily* child_family : child_families) {
      Int64 sub_dest_item_count = sbuf->getInt64();
      Int64UniqueArray sub_dest_uids;
      sub_dest_uids.resize(sub_dest_item_count);
      sbuf->getSpan(sub_dest_uids);
      IntegerUniqueArray parent_sub_dest_lids(sub_dest_item_count);
      itemFamily()->itemsUniqueIdToLocalId(parent_sub_dest_lids, sub_dest_uids, true);
      ItemVectorView parent_sub_dest_items(itemFamily()->view(parent_sub_dest_lids));
      // Temporary hack to find the associated sub-mesh
      DynamicMesh* dn = dynamic_cast<DynamicMesh*>(child_family->mesh());
      ARCANE_CHECK_POINTER(dn);
      dn->incrementalBuilder()->addParentItems(parent_sub_dest_items, child_family->itemKind());
    }
  }

  _applyDeserializePhase(IItemFamilySerializeStep::PH_Item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readAndAllocItemRelations()
{
  info(4) << "ItemsExchangeInfo2::readAndAllocItemRelations() " << itemFamily()->name();

  // The organization of the loops and the switch is not identical here to prepareToSend; for readability, they should be made similar.

  // Retrieves the info of the meshes of each receiver and creates the entities.
  for (Integer i = 0, is = m_exchanger->nbReceiver(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    info(5) << "Processing item message from " << comm->destination()
            << " for family " << itemFamily()->fullName();
    m_family_serializer->deserializeItemRelations(sbuf, &m_receive_local_ids[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readGroups()
{
  info(4) << "ItemsExchangeInfo2::readGroups() for "
          << m_item_family->name();

  Int64UniqueArray items_in_groups_uid;

  // Retrieves the info for the groups
  for (Integer i = 0, is = m_exchanger->nbReceiver(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();

    info(4) << "Processing group message from " << comm->destination()
            << " for family " << itemFamily()->fullName();

    // Checks for serialization errors.
    Integer magic_number = sbuf->getInteger();
    if (magic_number != GROUPS_MAGIC_NUMBER)
      ARCANE_FATAL("Internal error: bad magic number expected={0} found={1}",
                   GROUPS_MAGIC_NUMBER, magic_number);

    // Deserializes the groups
    for (Integer i_serializer = 0; i_serializer < m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->get(sbuf, items_in_groups_uid);
  }

  _applyDeserializePhase(IItemFamilySerializeStep::PH_Group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readVariables()
{
  info(4) << "ItemsExchangeInfo2::readVariables() for " << m_item_family->name();

  // Optionally resizes the data associated with variables.
  // NOTE GG: normally it seems to me that this is already done during
  // the call to DynamicMesh::_internalEndUpdateInit() in _exchangeItemsNew()
  // for all families.
  for (IItemFamily* family : m_families_to_exchange)
    family->_internalApi()->resizeVariables(true);

  _applyDeserializePhase(IItemFamilySerializeStep::PH_Variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
removeSentItems()
{
  // NOTE: This method is only called for particle families without ghosts.
  IItemFamily* family = itemFamily();
  IParticleFamily* pfamily = family->toParticleFamily();
  if (!pfamily)
    ARCANE_FATAL("This call is only valid for ParticleFamily. family={0}",
                 itemFamily()->name());
  if (pfamily->getEnableGhostItems())
    ARCANE_FATAL("This call is only valid for ParticleFamily without ghost",
                 itemFamily()->name());

  info(4) << "ItemsExchangeInfo2::removeSentItems(): " << family->name();

  for (Integer i = 0, is = m_exchanger->nbSender(); i < is; ++i) {
    ISerializeMessage* comm = m_exchanger->messageToSend(i);
    Int32 dest_rank = comm->destination().value();
    Int32ConstArrayView dest_items_local_id = m_send_local_ids[dest_rank];

    ItemVectorView dest_items = family->view(dest_items_local_id);

    //NOTE: (HP) Never tested on sub-meshes with particles
    IItemFamilyCollection child_families = itemFamily()->childFamilies();
    for (IItemFamily* child_family : child_families) {
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(family, child_family, dest_items);
      IParticleFamily* child_pfamily = child_family->toParticleFamily();
      ARCANE_CHECK_POINTER(child_pfamily);
      child_pfamily->removeParticles(sub_dest_items.view().localIds());
      child_family->endUpdate();
    }
    pfamily->removeParticles(dest_items_local_id);
  }
  family->endUpdate(); // Isn't this too strong because it also resizes the variables (but does it also handle groups vs partialEndUpdate)?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
processExchange()
{
  m_exchanger->processExchange(m_exchanger_option);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
finalizeExchange()
{
  for (IItemFamilySerializeStep* step : m_serialize_steps) {
    step->finalize();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
addSerializeStep(IItemFamilySerializeStep* step)
{
  m_serialize_steps.add(step);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
_applySerializeStep(IItemFamilySerializeStep::ePhase phase, const ItemFamilySerializeArgs& args)
{
  for (IItemFamilySerializeStep* step : m_serialize_steps) {
    if (step->phase() == phase)
      step->serialize(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
_applyDeserializePhase(IItemFamilySerializeStep::ePhase phase)
{
  for (IItemFamilySerializeStep* step : m_serialize_steps) {
    if (step->phase() != phase)
      continue;

    Integer nb_receive = m_exchanger->nbReceiver();
    {
      auto action = IItemFamilySerializeStep::eAction::AC_BeginReceive;
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action, nb_receive));
    }
    for (Integer i = 0; i < nb_receive; ++i) {
      ISerializeMessage* comm = m_exchanger->messageToReceive(i);
      ISerializer* sbuf = comm->serializer();
      Int32ConstArrayView local_ids = m_receive_local_ids[i].view();
      ItemFamilySerializeArgs serialize_args(sbuf, comm->destination().value(), local_ids, i);
      step->serialize(serialize_args);
    }
    {
      auto action = IItemFamilySerializeStep::eAction::AC_EndReceive;
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action, nb_receive));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
setParallelExchangerOption(const ParallelExchangerOptions& option)
{
  m_exchanger_option = option;
  m_exchanger->setVerbosityLevel(option.verbosityLevel());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
