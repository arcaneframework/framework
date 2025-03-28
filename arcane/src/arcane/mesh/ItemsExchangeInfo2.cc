// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsExchangeInfo2.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Echange des entités et leurs variables.                                   */
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

// TODO: a supprimer
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
  for( IItemFamilySerializeStep* step : m_serialize_steps )
    delete step;
  delete m_family_serializer;
  for(Integer i=0;i<m_groups_serializers.size(); ++i)
    delete m_groups_serializers[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void ItemsExchangeInfo2::
_addItemToSend(Int32 sub_domain_id,Item item)
{
  if (m_send_local_ids[sub_domain_id].empty())
    // Si premier élément, ajoute le sous-domaine à la liste des
    // sous-domaines communicants
    m_exchanger->addSender(sub_domain_id);
  m_send_local_ids[sub_domain_id].add(item.localId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemsExchangeInfo2::
computeExchangeInfos()
{
  // Détermine la liste des variables a échanger
  // On y intégre aussi les variables issues des familles enfants
  {
    m_families_to_exchange.add(itemFamily()); // La famille courante
    IItemFamilyCollection child_families = itemFamily()->childFamilies();
    for( IItemFamily* child_family : child_families )
      m_families_to_exchange.add(child_family);
   
    for( IItemFamily* current_family : m_families_to_exchange ){
      // Si la famille n'a pas de table de uniqueId, il ne faut pas
      // transferer les groupes car il n'est pas possible de convertir
      // les uniqueId en localId et le serialiseur en a besoin.
      // TODO: supprimer ce besoin dans le serialiseur
      if (current_family->hasUniqueIdMap())
        m_groups_serializers.add(new ItemGroupsSerializer2(current_family,m_exchanger.get()));
    }
  }

  for( IItemFamilySerializeStep* step : m_serialize_steps ){
    step->initialize();
  }

  bool r = m_exchanger->initializeCommunicationsMessages();
  m_receive_local_ids.resize(m_exchanger->nbReceiver());
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
setExchangeItems(ConstArrayView< std::set<Int32> > items_to_send)
{
  Int32 nb_part = m_item_family->mesh()->meshPartInfo().nbPart();
  m_send_local_ids.resize(nb_part);

  for( Integer i=0, is=items_to_send.size(); i<is; ++i ){
    Int64 n = items_to_send[i].size();
    if (n!=0){
      m_exchanger->addSender(i);
      //ItemInternalList items = m_item_family->itemsInternal();
      std::set<Int32>::const_iterator iids = items_to_send[i].begin();
      std::set<Int32>::const_iterator ids_end = items_to_send[i].end();
      //info() << "SEND kind=" << itemKindName(itemKind()) << " dest=" << i << " nb=" << n;
      for( ; iids!=ids_end; ++iids ){
        m_send_local_ids[i].add(*iids);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine la liste des entités à échanger.

 * \warning Cette méthode ne doit être utilisée que pour les familles
 * de particules.

 Cette opération se sert de la variable itemsOwner() et du champ
 owner() de chaque entité pour déterminer à qui chaque entité doit
 être envoyée. Par conséquent, il faut appeler cette opération
 avant que DynamicMesh::_setOwnerFromVariable() ne soit appelée.

 TODO: mettre cette méthode ailleurs.
*/
void ItemsExchangeInfo2::
computeExchangeItems()
{
  if (m_item_family->itemKind()!=IK_Particle)
    ARCANE_FATAL("This call is only valid for ParticleFamily. family={0}",
                 itemFamily()->name());

  Int32 nb_part = m_item_family->mesh()->meshPartInfo().nbPart();
  // Contient pour chaque sous-domaine la liste des entités à envoyer
  m_send_local_ids.resize(nb_part);
  // Liste des sous-domaines avec lesquels je dois communiquer.
  VariableItemInt32& items_owner(itemFamily()->itemsNewOwner());
    
  // Pour déterminer la liste des entités à envoyer, il suffit de comparer
  // le champs owner() de l'entité qui contient le propriétaire de l'entité courante
  // avec la variable itemsNewOwner() qui contient le nouveau propriétaire.
  // Si ces deux valeurs sont différentes, l'entité doit être envoyée.
  // ATTENTION: surtout, ne pas utiliser les ownItems(), car ceux ci sont calculés à la volée
  // (lazy evaluation)
  ENUMERATE_ITEM(i_item,itemFamily()->allItems()){
    Item item = *i_item;
    Int32 new_owner = items_owner[item];
    Int32 current_owner = item.owner();
    if (item.isOwn() && new_owner!=current_owner){
      _addItemToSend(new_owner,item);
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
  
  // Préparation des sérialiseurs de groupes
  for(Integer i_serializer=0;i_serializer<m_groups_serializers.size(); ++i_serializer){
    ItemGroupsSerializer2 * groups_serializer = m_groups_serializers[i_serializer];
    if (groups_serializer->itemFamily() == itemFamily()){
      // C'est la famille originale, donc sans transformation
      groups_serializer->prepareData(m_send_local_ids);
    }
    else{
      // C'est une sous-famille
      UniqueArray< SharedArray<Int32> > subitems_to_send(m_send_local_ids.size());
      for(Integer i_dest=0; i_dest<m_send_local_ids.size(); ++i_dest) {
        ItemVector subitems = MeshToMeshTransposer::transpose(itemFamily(), groups_serializer->itemFamily(), itemFamily()->view(m_send_local_ids[i_dest]));
        Int32Array & current_subitem_lids = subitems_to_send[i_dest];
        ENUMERATE_ITEM(iitem, subitems)
        if (iitem.localId() != NULL_ITEM_LOCAL_ID)
          current_subitem_lids.add(iitem.localId());
      }
      groups_serializer->prepareData(subitems_to_send);
    }
  }

  // Génère les infos pour chaque processeur à qui on va envoyer des entités
  ItemInfoListView items_internal(itemFamily());
  IItemFamilyCollection child_families = itemFamily()->childFamilies();

  const Integer nb_send = m_exchanger->nbSender();
  {
    auto action = IItemFamilySerializeStep::eAction::AC_BeginPrepareSend;
    for( IItemFamilySerializeStep* step : m_serialize_steps )
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action,nb_send));
  }

  for( Integer i=0; i<nb_send; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToSend(i);
    Int32 dest_sub_domain = comm->destination().value();
    // Liste des localId() des entités à envoyer
    Int32ConstArrayView dest_items_local_id = m_send_local_ids[dest_sub_domain];
    info(5) << "Processing message to " << dest_sub_domain
            << " for family " << itemFamily()->fullName();

    ISerializer* sbuf = comm->serializer();

    ItemFamilySerializeArgs serialize_args(sbuf,dest_sub_domain,dest_items_local_id,i);

    // Réserve la mémoire pour la sérialisation
    sbuf->setMode(ISerializer::ModeReserve);

    // Réserve pour les items et les uids des sous-items 
    m_family_serializer->serializeItems(sbuf,dest_items_local_id);
    m_family_serializer->serializeItemRelations(sbuf,dest_items_local_id);

    // Réserve pour les uids des sous-items (calcul en doublon de MeshToMeshTransposer::transpose avec les put)
    for( IItemFamily* child_family : child_families ) {
      ItemVectorView dest_items(items_internal, dest_items_local_id);
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(itemFamily(), child_family, dest_items);
      Integer sub_dest_item_count = 0;
      ENUMERATE_ITEM(iitem, sub_dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID)
          ++sub_dest_item_count;
      }
      sbuf->reserveInt64(1);
      sbuf->reserveSpan(eBasicDataType::Int64,sub_dest_item_count);
    }    

    _applySerializeStep(IItemFamilySerializeStep::PH_Item,serialize_args);

    sbuf->reserveInteger(1); // Pour nombre magique pour serialisation des groupes

    // Réserve pour les groupes
    for(Integer i_serializer=0; i_serializer<m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->serialize(serialize_args);
    
    _applySerializeStep(IItemFamilySerializeStep::PH_Group,serialize_args);

    // Les objets suivants sont désérialisés dans readVariables()
    
    // Réserve pour la sérialisation des variables
    _applySerializeStep(IItemFamilySerializeStep::PH_Variable,serialize_args);

    sbuf->allocateBuffer();

    // Sérialise les infos
    sbuf->setMode(ISerializer::ModePut);

    m_family_serializer->serializeItems(sbuf,dest_items_local_id);
    m_family_serializer->serializeItemRelations(sbuf,dest_items_local_id);

    // Sérialisation uids des sous-items (calcul en doublon de MeshToMeshTransposer::transpose avec les réserve)
    for( IItemFamily* child_family : child_families ) {
      ItemVectorView dest_items(items_internal, dest_items_local_id);
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(itemFamily(), child_family, dest_items);
      Int64UniqueArray sub_dest_uids;
      sub_dest_uids.reserve(sub_dest_items.size());
      ENUMERATE_ITEM(iitem, sub_dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID)
          sub_dest_uids.add(iitem->uniqueId());
      }
      sbuf->putInt64(sub_dest_uids.size());
      sbuf->putSpan(sub_dest_uids);
    }

    _applySerializeStep(IItemFamilySerializeStep::PH_Item,serialize_args);

    sbuf->put(GROUPS_MAGIC_NUMBER);

    // Sérialise la liste des groupes
    for(Integer i_serializer=0; i_serializer<m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->serialize(serialize_args);

    _applySerializeStep(IItemFamilySerializeStep::PH_Group,serialize_args);

    // Sérialise les infos pour les variables
    _applySerializeStep(IItemFamilySerializeStep::PH_Variable,serialize_args);
  }

  {
    auto action = IItemFamilySerializeStep::eAction::AC_EndPrepareSend;
    for( IItemFamilySerializeStep* step : m_serialize_steps )
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action,nb_send));
  }
}

void ItemsExchangeInfo2::
releaseBuffer()
{
  for( Integer i=0, is=m_exchanger->senderRanks().size(); i<is; ++i ) {
    ISerializeMessage* comm = m_exchanger->messageToSend(i);

    ISerializer* isbuf = comm->serializer();
    SerializeBuffer* sbuf = dynamic_cast<SerializeBuffer*>(isbuf);

    if(sbuf)
      sbuf->releaseBuffer();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readAndAllocItems()
{
  info(4) << "ItemsExchangeInfo2::readAndAllocItems() " << itemFamily()->name();

  // L'organisation des boucles et du switch n'est pas ici identiques à prepareToSend,
  // pour la lisibilité, il faudrait les rendre similaires

  // Récupère les infos des mailles de chaque receveur et créé les entités.
  for( Integer i=0, is=m_exchanger->nbReceiver(); i<is; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    info(5) << "Processing item message from " << comm->destination()
            << " for family " << itemFamily()->fullName();
    m_family_serializer->deserializeItems(sbuf,&m_receive_local_ids[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readAndAllocSubMeshItems()
{
  IItemFamilyCollection child_families = itemFamily()->childFamilies();
  for( Integer i=0, is=m_exchanger->nbReceiver(); i<is; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();

    for( IItemFamily* child_family : child_families ) {
      Int64 sub_dest_item_count = sbuf->getInt64();
      Int64UniqueArray sub_dest_uids;
      sub_dest_uids.resize(sub_dest_item_count);
      sbuf->getSpan(sub_dest_uids);
      IntegerUniqueArray parent_sub_dest_lids(sub_dest_item_count);
      itemFamily()->itemsUniqueIdToLocalId(parent_sub_dest_lids,sub_dest_uids,true);
      ItemVectorView parent_sub_dest_items(itemFamily()->view(parent_sub_dest_lids));
      // Hack temporaire pour trouver le sous-maillage associé
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
  info(4) <<  "ItemsExchangeInfo2::readAndAllocItemRelations() " << itemFamily()->name();

  // L'organisation des boucles et du switch n'est pas ici identiques à prepareToSend,
  // pour la lisibilité, il faudrait les rendre similaires

  // Récupère les infos des mailles de chaque receveur et créé les entités.
  for( Integer i=0, is=m_exchanger->nbReceiver(); i<is; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    info(5) << "Processing item message from " << comm->destination()
            << " for family " << itemFamily()->fullName();
    m_family_serializer->deserializeItemRelations(sbuf,&m_receive_local_ids[i]);
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

  // Récupère les infos pour les groupes
  for( Integer i=0, is=m_exchanger->nbReceiver(); i<is; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();

    info(4) << "Processing group message from " << comm->destination()
            << " for family " << itemFamily()->fullName();

    // Vérifie pas d'erreurs de sérialisation.
    Integer magic_number = sbuf->getInteger();
    if (magic_number!=GROUPS_MAGIC_NUMBER)
      ARCANE_FATAL("Internal error: bad magic number expected={0} found={1}",
                   GROUPS_MAGIC_NUMBER,magic_number);

    // Désérialise les groupes
    for(Integer i_serializer=0; i_serializer<m_groups_serializers.size(); ++i_serializer)
      m_groups_serializers[i_serializer]->get(sbuf,items_in_groups_uid);
  }

  _applyDeserializePhase(IItemFamilySerializeStep::PH_Group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
readVariables()
{
  info(4) << "ItemsExchangeInfo2::readVariables() for " << m_item_family->name();

  // Redimensionne éventuellement les données associées aux variables.
  // NOTE GG: normalement il me semble que c'est déjà fait lors
  // de l'appel à DynamicMesh::_internalEndUpdateInit() dans _exchangeItemsNew()
  // pour toutes les familles.
  for( IItemFamily* family : m_families_to_exchange )
    family->_internalApi()->resizeVariables(true);

  _applyDeserializePhase(IItemFamilySerializeStep::PH_Variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
removeSentItems()
{
  // NOTE: Cette méthode n'est appelée que pour les familles de particules
  // sans fantômes.
  IItemFamily* family = itemFamily();
  IParticleFamily* pfamily = family->toParticleFamily();
  if (!pfamily)
    ARCANE_FATAL("This call is only valid for ParticleFamily. family={0}",
                 itemFamily()->name());
  if (pfamily->getEnableGhostItems())
    ARCANE_FATAL("This call is only valid for ParticleFamily without ghost",
                 itemFamily()->name());

  info(4) << "ItemsExchangeInfo2::removeSentItems(): " << family->name();

  for( Integer i=0, is=m_exchanger->nbSender(); i<is; ++i ){
    ISerializeMessage* comm = m_exchanger->messageToSend(i);
    Int32 dest_rank = comm->destination().value();
    Int32ConstArrayView dest_items_local_id = m_send_local_ids[dest_rank];
    
    ItemVectorView dest_items = family->view(dest_items_local_id);

    //NOTE: (HP) Jamais testé sur des sous-maillages avec particules
    IItemFamilyCollection child_families = itemFamily()->childFamilies();
    for( IItemFamily* child_family : child_families){
      ItemVector sub_dest_items = MeshToMeshTransposer::transpose(family, child_family, dest_items);
      IParticleFamily* child_pfamily = child_family->toParticleFamily();
      ARCANE_CHECK_POINTER(child_pfamily);
      child_pfamily->removeParticles(sub_dest_items.view().localIds());
      child_family->endUpdate();
    }
    pfamily->removeParticles(dest_items_local_id);
  }
  family->endUpdate(); // N'est ce pas trop fort car ca resize aussi les variables (mais ca fait aussi les groupes vs partialEndUpdate) ?
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
  for( IItemFamilySerializeStep* step : m_serialize_steps ){
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
_applySerializeStep(IItemFamilySerializeStep::ePhase phase,const ItemFamilySerializeArgs& args)
{
  for( IItemFamilySerializeStep* step : m_serialize_steps ){
    if (step->phase()==phase)
      step->serialize(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemsExchangeInfo2::
_applyDeserializePhase(IItemFamilySerializeStep::ePhase phase)
{
  for( IItemFamilySerializeStep* step : m_serialize_steps ){
    if (step->phase()!=phase)
      continue;

    Integer nb_receive = m_exchanger->nbReceiver();
    {
      auto action = IItemFamilySerializeStep::eAction::AC_BeginReceive;
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action,nb_receive));
    }
    for( Integer i=0; i<nb_receive; ++i ){
      ISerializeMessage* comm = m_exchanger->messageToReceive(i);
      ISerializer* sbuf = comm->serializer();
      Int32ConstArrayView local_ids = m_receive_local_ids[i].view();
      ItemFamilySerializeArgs serialize_args(sbuf,comm->destination().value(),local_ids,i);
      step->serialize(serialize_args);
    }
    {
      auto action = IItemFamilySerializeStep::eAction::AC_EndReceive;
      step->notifyAction(IItemFamilySerializeStep::NotifyActionArgs(action,nb_receive));
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

