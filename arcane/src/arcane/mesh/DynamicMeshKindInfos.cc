// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshKindInfos.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Infos de maillage pour un genre d'entité donnée.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IItemInternalSortFunction.h"
#include "arcane/core/ItemFamilyCompactInfos.h"
#include "arcane/core/IMeshCompacter.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/ItemFamilyItemListChangedEventArgs.h"

#include "arcane/mesh/DynamicMeshKindInfos.h"
#include "arcane/mesh/ItemFamily.h"

#include <algorithm>
#include <set>

// #define ARCANE_DEBUG_MESH

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshKindInfos::
DynamicMeshKindInfos(IMesh* mesh,eItemKind kind,const String& kind_name)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_item_family(nullptr)
, m_kind(kind)
, m_kind_name(kind_name)
, m_nb_item(0)
, m_is_verbose(false)
, m_use_new_finalize(true)
, m_is_first_finalize(true)
, m_has_unique_id_map(true)
, m_compact_infos(nullptr)
, m_item_internals_buffer(new MultiBufferT<ItemInternal>(5000))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshKindInfos::
~DynamicMeshKindInfos()
{
  delete m_item_internals_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
build()
{
  // Créé le groupe contenant toutes les entités de ce genre
  {
    // Si le nom de ce groupe change, il faut mettre a jour 
    // le nom equivalent dans VariableInfo.cc
    StringBuilder str("All");
    str += m_kind_name;
    str += "s";
    m_all_group_name = str.toString();
    m_all_group = m_item_family->createGroup(m_all_group_name);
    m_all_group.internal()->setIsAllItems();
    m_all_group.setLocalToSubDomain(true);
  }
  if (!platform::getEnvironmentVariable("ARCANE_OLD_FINALIZE").null()){
    m_use_new_finalize = false;
    pwarning() << "USE OLD FINALIZE";
  }
  m_is_first_finalize = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
void DynamicMeshKindInfos::
finalizeMeshChanged()
{
  Trace::Setter mci(traceMng(),"Mesh");

  ItemGroupImpl& impl = *m_all_group.internal();
  Integer internal_size =  m_internals.size();

  {
    info(5) << A_FUNCNAME << " " << m_kind_name
            << " nb_item=" << m_nb_item
            << " internal size= " << internal_size
            << " max_local_id=" << maxUsedLocalId();
#if 0
    for( Integer i=0; i<internal_size; ++i ){
      ItemInternal* item = m_internals[i];
      info() << "Item " << i << " lib=" << item->localId()
             << " uid=" << item->uniqueId() << " owner=" << item->owner()
             << " item=" << item
             << " removed?=" << item->isSuppressed();
    }
#endif

  }
      
  Integer nb_added_items = m_added_items.size();
  Integer nb_removed_items = m_removed_items.size();
  bool test_mode = m_use_new_finalize;

  info(5) << A_FUNCNAME << " " << m_kind_name
          << " nb_item " << m_nb_item
          << " internal size " << internal_size
          << " nb_add=" << nb_added_items
          << " nb_remove=" << nb_removed_items;

  // GG: passage en OLD le 11/2018. Si tout est OK du côté IFPEN
  // on pourra définitivement supprimer cela.
#if OLD
  IParallelMng* pm = m_mesh->parallelMng();
  // En séquentiel, positionne le champs m_owner de chaque entité.
  // TODO: ne doit pas être fait ici, mais dans le lecteur de maillage...
  if (!pm->isParallel() && nb_added_items!=0){
    const Integer sid = pm->commRank();
    for( Integer i=0; i<nb_added_items; ++i )
      m_internals[ m_added_items[i] ]->setOwner(sid,sid);
  }
#endif
  // (HP) TODO: On peut surement optimiser la réorganisation via des 
  // addItems, removeItems, changeIds...
  // Le problème est qu'ici il ne faut pas impacter les sous-groupes calculés
  impl.beginTransaction();
  if (nb_added_items==0 && nb_removed_items!=0 && test_mode){
    impl.removeItems(m_removed_items,true);
  } else if (nb_added_items!=0 && nb_removed_items==0 && test_mode){
    impl.addItems(m_added_items,true); 
  } else {
    impl.removeAddItems(m_removed_items,m_added_items,true); 
  }
  impl.endTransaction();
  
  // restore integrity of allItems group after this agressive modification
  impl.checkNeedUpdate();

#if defined(ARCANE_DEBUG_MESH)
  if (arcaneIsCheck())
    m_all_group.checkValid();
#endif
  // Il faut aussi changer les groupes des entités propres
  // (A faire uniquement en parallele ?)
  m_item_family->notifyItemsOwnerChanged();

#if defined(ARCANE_DEBUG_MESH)
  if (arcaneIsCheck())
    checkValid();
#endif

  m_removed_items.clear();
  m_added_items.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
itemsUniqueIdToLocalId(ArrayView<Int64> ids,bool do_fatal) const
{
  if (!m_has_unique_id_map)
    _badUniqueIdMap();
  if (!arcaneIsCheck()){
    if (do_fatal){
      for( Integer i=0, s=ids.size(); i<s; ++i ){
        Int64 unique_id = ids[i];
        ids[i] = m_items_map.findLocalId(unique_id);
      }
    }
    else{
      for( Integer i=0, s=ids.size(); i<s; ++i ){
        Int64 unique_id = ids[i];
        ids[i] = m_items_map.tryFindLocalId(unique_id);
      }
    }
  }
  else{
    Integer nb_error = 0;
    for( Integer i=0, s=ids.size(); i<s; ++i ){
      Int64 unique_id = ids[i];
      ids[i] = m_items_map.tryFindLocalId(unique_id);
      if ((ids[i] == NULL_ITEM_LOCAL_ID) && do_fatal) {
        if (nb_error<10){
          error() << "DynamicMeshKindInfos::itemsUniqueIdToLocalId() can't find "
                  << "entity " << m_kind_name << " with global id "
                  << unique_id << " in the subdomain.";
        }
        ++nb_error;
      }
    }
    if (nb_error!=0){
      if (do_fatal)
        ARCANE_FATAL("{0} entities not found",nb_error);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       Int64ConstArrayView unique_ids,bool do_fatal) const
{
  if (!m_has_unique_id_map)
    _badUniqueIdMap();
  if (!arcaneIsCheck()){
    if (do_fatal){
      for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
        Int64 unique_id = unique_ids[i];
        local_ids[i] = (unique_id == NULL_ITEM_UNIQUE_ID) ? NULL_ITEM_LOCAL_ID : m_items_map.findLocalId(unique_id);
      }
    }
    else{
      for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
        Int64 unique_id = unique_ids[i];
        local_ids[i] = m_items_map.tryFindLocalId(unique_id);
      }
    }
  }
  else{
    Integer nb_error = 0;
    for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
      Int64 unique_id = unique_ids[i];
      local_ids[i] = m_items_map.tryFindLocalId(unique_id);
      if ((local_ids[i] == NULL_ITEM_LOCAL_ID) && do_fatal && unique_id != NULL_ITEM_UNIQUE_ID) {
        if (nb_error<10){
          error() << "DynamicMeshKindInfos::itemsUniqueIdToLocalId() can't find "
                  << "entity " << m_kind_name << " with global id "
                  << unique_id << " in the subdomain.";
        }
        ++nb_error;
      }
    }
    if (nb_error!=0){
      if (do_fatal)
        ARCANE_FATAL("{0} entities not found",nb_error);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                       ConstArrayView<ItemUniqueId> unique_ids,bool do_fatal) const
{
  if (!m_has_unique_id_map)
    _badUniqueIdMap();
  if (!arcaneIsCheck()){
    if (do_fatal){
      for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
        Int64 unique_id = unique_ids[i];
        local_ids[i] = m_items_map.findLocalId(unique_id);
      }
    }
    else{
      for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
        Int64 unique_id = unique_ids[i];
        local_ids[i] = m_items_map.tryFindLocalId(unique_id);
      }
    }
  }
  else{
    Integer nb_error = 0;
    for( Integer i=0, s=unique_ids.size(); i<s; ++i ){
      Int64 unique_id = unique_ids[i];
      local_ids[i] = m_items_map.tryFindLocalId(unique_id);
      if ((local_ids[i] == NULL_ITEM_LOCAL_ID) && do_fatal) {
        if (nb_error<10){
          error() << "DynamicMeshKindInfos::itemsUniqueIdToLocalId() can't find "
                  << "entity " << m_kind_name << " with global id "
                  << unique_id << " in the subdomain.";
        }
        ++nb_error;
      }
    }
    if (nb_error!=0){
      if (do_fatal)
        ARCANE_FATAL("{0} entities not found",nb_error);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
prepareForDump()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
readFromDump()
{
  m_all_group = m_mesh->findGroup(m_all_group_name);
  // Supprime toutes les entités
  m_nb_item = 0;
  m_internals.clear();
  _updateItemSharedInfoInternalView();
  m_free_internals.clear();
  m_items_map.clear();
  delete m_item_internals_buffer;
  m_item_internals_buffer = new MultiBufferT<ItemInternal>(5000);
  m_free_internals_in_multi_buffer.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
checkValid()
{
  Int32 sid = m_mesh->meshPartInfo().partRank();
  
  debug() << "DynamicMeshKindInfos::checkValid(): " << m_kind_name;

  Integer nb_killed_item = 0;
  Integer nb_internal = m_internals.size();
  Integer nb_error = 0;

  // Premièrement, le item->localId() doit correspondre à l'indice
  // dans le tableau m_internal
  for( Integer i=0; i<nb_internal; ++i ){
    ItemInternal* item = m_internals[i];
    if (item->localId()!=i && nb_error<10){
      error() << "The local id (" << item->localId() << ") of the entity "
              << m_kind_name << ':' << item->uniqueId() << " is not "
              << "consistent with its internal value (" << i << ")";
      ++nb_error;
    }
    if (item->isSuppressed()){
      ++nb_killed_item;
      continue;
    }
    if (item->owner()==NULL_SUB_DOMAIN_ID && nb_error<10){
      error() << "entity " << m_kind_name << ":lid=" << item->localId()
              << ":uid=" << item->uniqueId() << "  belongs to no subdomain";
      ++nb_error;
    }

    if (item->owner()!=sid && item->isOwn() && nb_error<10){
      error() << "entity " << m_kind_name << ":lid=" << item->localId()
              << ":uid=" << item->uniqueId() << "  incoherence between isOwn() and owner()"
              << " " << item->owner() << ' ' << sid;
      ++nb_error;
    }

  }
  Integer nb_free = m_free_internals.size();
  if (nb_killed_item!=nb_free){
    error() << "DynamicMeshKindInfos::checkValid(): " << m_kind_name
            << ": incoherence between killed and free entities"
            << " free=" << nb_free
            << " killed=" << nb_killed_item
            << " internal=" << nb_internal
            << " count=" << m_nb_item;
    ++nb_error;
  }
  
  if (nb_error!=0)
    ARCANE_FATAL("Internal error in the mesh structure mesh={0} part={1}",
                 m_mesh->name(),sid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef OLD
class ItemParticleCompareWithSuppression
{
 public:
  bool operator()(const ItemInternal* item1,const ItemInternal* item2) const
  {
    // Il faut mettre les entités détruites en fin de liste
    //cout << "Compare: " << item1->uniqueId() << " " << item2->uniqueId() << '\n';
    bool s1 = item1->isSuppressed();
    bool s2 = item2->isSuppressed();
    if (s1 && !s2)
      return false;
    if (!s1 && s2)
      return true;
    Int64 uid1 = Particle(item1).cell(0)->uniqueId();
    Int64 uid2 = Particle(item2).cell(0)->uniqueId();
    if (uid1==uid2)
      return item1->uniqueId() < item2->uniqueId();
    return uid1<uid2;
  }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemLocalIdAndUniqueId
{
 public:
  bool operator()(const ItemLocalIdAndUniqueId& item1,const ItemLocalIdAndUniqueId item2) const
    {
      return item1.m_unique_id < item2.m_unique_id;
    }
  Integer m_local_id;
  Integer m_unique_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView DynamicMeshKindInfos::
oldToNewLocalIds() const
{
  if (!m_compact_infos)
    return Int32ConstArrayView();
  return m_compact_infos->oldToNewLocalIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView DynamicMeshKindInfos::
newToOldLocalIds() const
{
  if (!m_compact_infos)
    return Int32ConstArrayView();
  return m_compact_infos->newToOldLocalIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcule les nouveaux id locaux des entités pour qu'ils soient consécutifs.
    
  Si \a do_sort est vrai, les entités sont triées de telle sorte que leur id unique
  et id local soient dans un ordre croissant.
    
  Avant appel à cette fonction, le maillage doit être valide et finalisé

  Après appel à cette fonction, les tableaux oldToNewLocalIds() et
  newToOldLocalIds() sont renseignés et
  contiennent pour chaque entité les conversions entre nouveaux et
  anciens numéros locaux.
*/
void DynamicMeshKindInfos::
beginCompactItems(ItemFamilyCompactInfos& compact_infos)
{
  m_compact_infos = &compact_infos;
  bool do_sort = compact_infos.compacter()->isSorted();

  if (arcaneIsCheck())
    checkValid();

  Integer nb_internal = m_internals.size();
  info(4) << m_kind_name << " beginCompactItems() Family Internal Size=" << nb_internal
          << " max_id=" << maxUsedLocalId()
          << " nb_item=" << m_nb_item
          << " do_sort=" << do_sort;

  if (nb_internal==0)
    return;

  UniqueArray<Int32> old_to_new_local_ids(maxUsedLocalId());
  UniqueArray<Int32> new_to_old_local_ids(m_nb_item);
  // m_new_to_old_local_ids[lid] contient l'ancien numéro local de l'entité dont le nouveau numéro local est \a lid
  // m_old_to_new_local_ids[lid] contient le nouveau numéro local de l'entité dont l'ancien numéro local est \a lid
  old_to_new_local_ids.fill(NULL_ITEM_LOCAL_ID);
  // après compactage, il ne peut y avoir que m_nb_item items actifs (et forcément en tête de numérotation)
  new_to_old_local_ids.fill(NULL_ITEM_LOCAL_ID);

  if (do_sort){
    // Fait une copie temporaire des ItemInternal pour le tri
    UniqueArray<ItemInternal*> items(m_internals);

    debug() << "BEGIN = " << items.data() << " " << m_internals.data()
            << " S1=" << items.size() << " S2=" << m_internals.size();
    const bool print_infos = false;
    {
      ITimerMng* tm = m_mesh->parallelMng()->timerMng();
      Timer timer(tm,"DynamicMeshKindInfos::beginCompactItems()",Timer::TimerReal);
      IItemInternalSortFunction* sort_func = m_item_family->itemSortFunction();
      if (print_infos){
        for( Integer i=0, is=items.size(); i<is; ++i ){
          info() << "Before Sort: " << i << " uid=" << items[i]->uniqueId()
                 << " lid=" << items[i]->localId()
                 << " destroyed?=" << items[i]->isSuppressed();
        }
      }
      {
        Timer::Sentry sentry(&timer);
        sort_func->sortItems(items);
      }
      info(4) << "Temps pour trier les entités <" << m_kind_name << "> "
              << timer.lastActivationTime()
              << " sort_func_name=" << sort_func->name();
      info(4) << "Sort Infos: size=" << items.size() << " nb_item=" << m_nb_item;
      if (print_infos){
        for( Integer i=0, n=items.size(); i<n; ++i ){
          info() << "After Sort: " << i << " uid=" << items[i]->uniqueId()
                 << " lid=" << items[i]->localId()
                 << " destroyed?=" << items[i]->isSuppressed();
        }
      }
    }

#ifdef ARCANE_TEST_ADD_MESH
    typedef std::set<ItemInternal*,ItemCompare> ItemSet;
    ItemSet items_set;
    items_set.insert(m_internals.begin(),m_internals.end());
    UniqueArray<ItemInternal*> items(m_internals.size());
    {
      ItemSet::const_iterator b = items_set.begin();
      ItemSet::const_iterator e = items_set.end();
      for( Integer z=0; b!=e; ++b, ++z)
        items[z] = *b;
    }
#endif

    for( Integer i=0, n=m_nb_item; i<n; ++i ){
      Integer current_local_id = items[i]->localId();
      //info() << "Item: " << " uid=" << items[i]->uniqueId() << " lid="
      //<< current_local_id << " newid=" << i;
      old_to_new_local_ids[ current_local_id ] = i;
      new_to_old_local_ids[ i ] = current_local_id;
    }
  }
  else{
    const bool use_old = true;
    if (use_old){
      info(4) << "USE_OLD_METHOD";
      Int32ConstArrayView existing_items = m_all_group.internal()->itemsLocalId();

      for( Integer i=0; i<m_nb_item; ++i ){
        Integer current_local_id = existing_items[i];
        old_to_new_local_ids[ current_local_id ] = i;
        new_to_old_local_ids[ i ] = current_local_id;
      }

      // Place les éléments détruits en fin de liste et vérifie
      // que le nombre de trous + le nombre d'entités est égal à #nb_internal
      {
        Integer old_index = m_nb_item;
        Integer total_nb = maxUsedLocalId();
        for( Integer i=0, n=total_nb; i<n; ++i ){
          //bool is_destroyed = false;
          if (old_to_new_local_ids[i]==NULL_ITEM_ID){
            //is_destroyed = true;
            old_to_new_local_ids[i] = old_index;
            //m_new_to_old_local_ids[old_index] = i;
            ++old_index;
          }
          //debug() << "OLD_TO_NEW i=" << i
          //        << " old=" << m_old_to_new_local_ids[i]
          //        << " destroyed=" << is_destroyed;
        }
        if (old_index!=total_nb){
          ARCANE_FATAL("Family '{0}' bad indices: expected={1} found={2} (nb_internal={3})",
                       itemFamily()->name(),old_index,total_nb,nb_internal);
        }
      }
    }
    else{
      // Compactage sans triage, ce qui revient à déplacer les trous à la fin

      // Place les éléments détruits en fin de liste et vérifie
      // que l'on a bien tout rangé (on conserve l'ordre relatif des items)
      Integer new_index = 0;
      Integer free_index = m_nb_item;
      for( Integer i=0, n=maxUsedLocalId(); i<n; ++i ){
        if (m_internals[i]->isSuppressed()) {
          old_to_new_local_ids[i] = free_index;
          // m_new_to_old_local_ids[free_index] = i; // non utilisé quand m_new_to_old_local_ids restreint à m_nb_item
          ++free_index;
        } else {
          old_to_new_local_ids[i] = new_index;
          new_to_old_local_ids[new_index] = i;
          ++new_index;
        }
        //debug() << "OLD_TO_NEW i=" << i
        //        << " old=" << m_old_to_new_local_ids[i]
        //        << " destroyed=" << m_internals[i]->isSuppressed();
      }
      if (new_index!=m_nb_item){
          ARCANE_FATAL("Family '{0}' bad indices: expected={1} found={2}",
                       itemFamily()->name(),m_nb_item,new_index);
      }
    }
  }
  compact_infos.setOldToNewLocalIds(std::move(old_to_new_local_ids));
  compact_infos.setNewToOldLocalIds(std::move(new_to_old_local_ids));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
finishCompactItems(ItemFamilyCompactInfos& compact_infos)
{
  if (arcaneIsCheck())
    checkValid();

  Int32ConstArrayView old_to_new_local_ids = compact_infos.oldToNewLocalIds();
  Int32ConstArrayView new_to_old_local_ids = compact_infos.newToOldLocalIds();

  Integer nb_internal = m_internals.size();
  
  info(4) << m_kind_name << " Family Compression: nb_item=" << m_nb_item
          << " nb_internal=" << nb_internal
          << " buf_size=" << m_item_internals_buffer->bufferSize()
          << " nb_allocated_buf=" << m_item_internals_buffer->nbAllocatedBuffer()
          << " nb_free" << m_free_internals_in_multi_buffer.size();

  if (nb_internal==0)
    return;

  // Finalise la réorganisation des structures après un tri
  // 1. Il faut mettre à jour la structure m_items_map pour référencer
  // le nouveau local_id
  // 2. Il faut recopier les valeurs de chaque ItemInternal pour qu'il
  // soit bien placé dans la nouvelle numérotation.
  // IMPORTANT: Cette opération doit toujours être la dernière car ensuite
  // on perd la relation entre les anciens local_ids et les nouveaux à
  // travers cette structure
  m_items_map._changeLocalIds(m_internals, old_to_new_local_ids);

  if (m_is_verbose){
    info() << "DumpItemsBefore:";
    _dumpList();
  }

  // Pour l'instant, utilise un tableau temporaire
  // TODO: tableau a supprimer
  UniqueArray<ItemInternal> new_items(m_nb_item);
  UniqueArray<Int64> new_uids(m_nb_item);
  UniqueArray<Int32> new_owners(m_nb_item);
  UniqueArray<Int32> new_flags(m_nb_item);
  UniqueArray<Int16> new_typeids(m_nb_item);

  for( Integer i=0, n=m_nb_item; i<n; ++i ) {
    ItemInternal* old_item = m_internals[ new_to_old_local_ids[ i ] ];
    new_uids[i] = old_item->uniqueId().asInt64();
    new_owners[i] = old_item->owner();
    new_flags[i] = old_item->flags();
    new_typeids[i] = old_item->typeId();
    new_items[i] = *old_item;
  }

  Integer nb_error = 0;
  Int32 sid = m_mesh->meshPartInfo().partRank();

  for( Integer i=0; i<m_nb_item; ++i ){
    ItemInternal* ii = m_internals[i];
    *ii = new_items[i];
    ii->setLocalId(i);
    ii->setFlags(new_flags[i]);
    ii->setOwner(new_owners[i],sid);
    ii->setUniqueId(new_uids[i]);
    ItemSharedInfo* isi = ItemInternalCompatibility::_getSharedInfo(ii);
    ii->_setSharedInfo(isi,ItemTypeId(new_typeids[i]));
    // L'entité est marqué comme créée
    //_setAdded(ii);
#ifdef ARCANE_CHECK
    if (ii->isSuppressed()){
      if (nb_error<10)
        error() << "Entity deleted from the list of created entities "
                << ItemPrinter(ii) << " index=" << i;
      ++nb_error;
    }
#endif /* ARCANE_CHECK */
  }

  // Il faut remplir les m_free_internals_in_multi_buffer avec les entités supprimées
  // pour qu'elles puissent être réutilisées sinon on va réallouer
  // des blocs dans m_item_internals_buffer à chaque recréation des
  // entités
  for( Integer i=m_nb_item, n=m_internals.size(); i<n; ++i ){
    // Empile dans l'ordre inverse pour que allocOne()
    // dépile dans l'ordre des m_internals et profite des
    // effets de cache.
    ItemInternal* removed_item = m_internals[m_nb_item + n-(i+1)];
    m_free_internals_in_multi_buffer.add(removed_item);
  }

  // Retaillage optimal des tableaux
  m_internals.resize(m_nb_item);
  _updateItemSharedInfoInternalView();
  m_free_internals.clear();

  if (m_is_verbose){
    info() << "DumpItemsAfter:";
    _dumpList();
  }

  if (nb_error!=0)
    ARCANE_FATAL("Error in compacting nb_error={0}",nb_error);

  debug() << "Compression: old=" << nb_internal << " new=" << m_nb_item
          << " internal=" << m_internals.size()
          << " free=" << m_free_internals.size();

  // Vide les tableaux de 'compact_infos' car ils ne sont plus valides.
  compact_infos.clear();

  m_compact_infos = nullptr;
  finalizeMeshChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace{
  class _Inverse
  {
   public:
    bool operator()(Int32 i1,Int32 i2) const
    {
      return i1>i2;
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
clear()
{
  Int32ConstArrayView items_id = m_all_group.internal()->itemsLocalId();
  
  // Pour réutiliser au moins les m_free_internals afin
  // d'avoir des localId() consécutifs, il faut trier
  // les ids par ordre décroissant.
  Int32UniqueArray ids(items_id);
  std::sort(std::begin(ids),std::end(ids),_Inverse());

  removeMany(ids);
  //printFreeInternals(10000);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
printFreeInternals(Integer max_print)
{
  Integer nb_free = m_free_internals.size();
  max_print = math::min(nb_free,max_print);
  for( Integer i=0; i<max_print; ++i ){
    Integer pos = (nb_free-1)-i;
    Int32 index = m_free_internals[pos];
    ItemInternal* iitem = m_internals[index];
    info() << "I=" << i << " pos=" << pos << " index=" << index
           << " local_id=" << iitem->localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
removeMany(Int32ConstArrayView local_ids)
{
  Int64ArrayView uids = *(m_item_family->uniqueIds());
  Integer nb_item = local_ids.size();

  if (m_has_unique_id_map){
    for( Integer i=0; i<nb_item; ++i ){
      Int32 lid = local_ids[i];
      ItemInternal* item = m_internals[lid];
#ifdef ARCANE_CHECK
      _checkActiveItem(item);
#endif
      _setSuppressed(item);
      if (uids[lid] != NULL_ITEM_UNIQUE_ID)
        m_items_map.remove(uids[lid]);
    }
  }
  else{
    for( Integer i=0; i<nb_item; ++i ){
      Int32 lid = local_ids[i];
      ItemInternal* item = m_internals[lid];
#ifdef ARCANE_CHECK
      _checkActiveItem(item);
#endif
      _setSuppressed(item);
    }
  }

  m_nb_item -= nb_item;
  Integer nb_removed = m_removed_items.size();
  m_removed_items.resize(nb_removed + nb_item);
  memcpy(m_removed_items.data()+nb_removed,local_ids.data(),sizeof(Int32)*nb_item);
  Integer nb_free = m_free_internals.size();
  m_free_internals.resize(nb_free + nb_item);
  memcpy(m_free_internals.data()+nb_free,local_ids.data(),sizeof(Int32)*nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
setHasUniqueIdMap(bool v)
{
  if (m_nb_item!=0)
    ARCANE_FATAL("family is not empty");
  m_has_unique_id_map = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_checkActiveItem(ItemInternal* item)
{
  if (item->isSuppressed())
    ARCANE_FATAL("Attempting to remove an entity already deleted item={0}",
                 ItemPrinter(item));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_dumpList()
{
  Integer nb_internal = m_internals.size();
  Integer computed_nb_item = 0;
  for( Integer i=0; i<nb_internal; ++i ){
    ItemInternal* item = m_internals[i];
    bool is_suppressed = item->isSuppressed();
    if (is_suppressed)
      ++computed_nb_item;
    info() << "Item: INDEX=" << i
           << " LID=" << item->localId()
           << " UID=" << item->uniqueId()
           << " KILLED=" << is_suppressed;
  }
  info() << "EndOfDump: "
         << " nb_internal=" << nb_internal
         << " nb_item=" << m_nb_item
         << " computed_nb_item= " << (nb_internal-computed_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_badSameUniqueId(Int64 unique_id) const
{
  ARCANE_FATAL("duplicate unique id family={0} uid={1}",
               m_item_family->name(),unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_badUniqueIdMap() const
{
  if (m_has_unique_id_map)
    ARCANE_FATAL("family have unique id map");
  else
    ARCANE_FATAL("family does not have unique id map");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
setItemFamily(ItemFamily* item_family)
{
  m_item_family = item_family;
  m_common_item_shared_info = item_family->commonItemSharedInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_updateItemSharedInfoInternalView()
{
  if (m_common_item_shared_info)
    m_common_item_shared_info->m_items_internal = m_internals.constView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObservableView<const ItemFamilyItemListChangedEventArgs&> DynamicMeshKindInfos::
itemListChangedEvent()
{
  return EventObservableView<const ItemFamilyItemListChangedEventArgs&>(m_item_list_change_event);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_notifyRemove2(ItemInternal* item)
{
  ItemFamilyItemListChangedEventArgs args(m_item_family,item->localId(),item->uniqueId());
  m_item_list_change_event.notify(args);
  args.setIsAdd(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshKindInfos::
_notifyAdd2(ItemInternal* item,Int64 uid)
{
  ItemFamilyItemListChangedEventArgs args(m_item_family,item->localId(),uid);
  args.setIsAdd(true);
  m_item_list_change_event.notify(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

