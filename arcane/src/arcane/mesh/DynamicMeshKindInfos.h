// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshKindInfos.h                                      (C) 2000-2025 */
/*                                                                           */
/* Infos de maillage pour un genre d'entité donnée.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHKINDINFOS_H
#define ARCANE_MESH_DYNAMICMESHKINDINFOS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Event.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/VariableTypedef.h"

#include "arcane/mesh/ItemInternalMap.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Define pour désactiver les évènements si on souhaite tester
// l'influence sur les performances (a priori il n'y en a pas).
#define ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos de maillage pour un genre donné d'entité.
 *
 * Une instance de cette classe gère toutes les structures de maillage
 * pour une entité d'un genre donné.
 */
class ARCANE_MESH_EXPORT DynamicMeshKindInfos
: public TraceAccessor
{
 public:

  // TODO: a supprimer
  typedef Arcane::mesh::ItemInternalMap ItemInternalMap;

 private:

  using ItemInternalMapData = ItemInternalMap::BaseData;

 public:

  //! Créé une instance pour un maillage et un genre donnée.
  DynamicMeshKindInfos(IMesh* mesh,eItemKind kind,const String& kind_name);
  //! Libère les ressources
  ~DynamicMeshKindInfos();

 public:

  void build();
  
  //! Réalloue et recalcule les structures après modification du maillage
  /*! @internal En particulier injecte les added et removed items dans le groupe total
   *  all_items courant. Les changements sont faits en direct sur ce groupe sans protection.
   */
  void finalizeMeshChanged();

  /*! \brief Numéro local le plus grand utilisé.
    
    Ce numéro est utilisé pour allouer les tableaux des variables
    sur les entités du maillage.
  */
  Integer maxUsedLocalId() const { return m_internals.size(); }

  //! Prépare les variables pour les sauvegardes
  void prepareForDump();

  //! Restaure les infos à partir des dumps
  void readFromDump();

  //! Groupe de toutes les entités
  ItemGroup allItems() const { return m_all_group; }
  
  //! Liste des entitées ajoutées ou retirées depuis le dernier endUpdate()
  Int32ConstArrayView addedItems  () const { return m_added_items;  }
  Int32ConstArrayView removedItems() const { return m_removed_items;}

  //! Liste interne des ItemInternal's
  /*! En lecture seule, la version en écriture a été supprimée comme indiqué en todo */
  ConstArrayView<ItemInternal*> itemsInternal() const { return m_internals; }

  ItemInternal* itemInternal(Int32 local_id) const { return m_internals[local_id]; }

  //! Ajoute une entité de numéro unique \a unique_id
  ItemInternal* allocOne(Int64 unique_id)
  {
    bool need_alloc = false;
    ItemInternal* next = _allocOne(need_alloc);
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyAdd(next,unique_id);
#endif
    if (m_has_unique_id_map)
      if (!m_items_map.add(unique_id,next))
        _badSameUniqueId(unique_id);
    return next;
  }

  //! Ajoute une entité de numéro unique \a unique_id
  ItemInternal* allocOne(Int64 unique_id,bool& need_alloc)
  {
    ItemInternal* next = _allocOne(need_alloc);
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyAdd(next,unique_id);
#endif
    if (m_has_unique_id_map)
      if (!m_items_map.add(unique_id,next))
        _badSameUniqueId(unique_id);
    return next;
  }

  //! Supprime l'entité \a item
  void removeOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    _setSuppressed(item);
    if (m_has_unique_id_map){
      Int64 uid = item->uniqueId().asInt64();
      if (uid != NULL_ITEM_UNIQUE_ID)
        m_items_map.remove(uid);
    }
    m_free_internals.add(item->localId());
    m_removed_items.add(item->localId());
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyRemove(item);
#endif
    --m_nb_item;
  }

  //! Supprime l'entité détachée \a item
  void removeDetachedOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    _setSuppressed(item);
#ifndef REMOVE_UID_ON_DETACH
    if (m_has_unique_id_map)
      m_items_map.remove(item->uniqueId().asInt64());
#endif
    m_free_internals.add(item->localId());
    m_removed_items.add(item->localId());
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
    _notifyRemove(item);
#endif
    --m_nb_item;
  }

  /*!
   * \brief Détache une l'entité \a item.
   *
   * L'entité est supprimée de la liste des uniqueId()
   * si la macro REMOVE_UID_ON_DETACH est définie
   */  
  void detachOne(ItemInternal* item)
  {
#ifdef ARCANE_CHECK
    _checkActiveItem(item);
#endif
    // Peut-être faut-il la marquer supprimée.
    //_setSuppressed(item);
#ifdef REMOVE_UID_ON_DETACH
    if (m_has_unique_id_map)
      m_items_map.remove(item->uniqueId().asInt64());
#endif /* REMOVE_UID_ON_DETACH */
    item->setDetached(true);
  }

  //! Supprime une liste d'entités
  void removeMany(Int32ConstArrayView local_ids);

  //! Recherche l'entité de numéro unique \a unique_id et la créé si elle n'existe pas
  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc)
  {
#ifdef ARCANE_CHECK
    if (!m_has_unique_id_map)
      _badUniqueIdMap();
#endif
    ItemInternalMap::LookupData item_data = m_items_map._lookupAdd(uid, 0, is_alloc);
    if (is_alloc){
      bool need_alloc;
      item_data.setValue(_allocOne(need_alloc));
#ifdef ARCANE_ENABLE_EVENT_FOR_DYNAMICMESHKINDINFO
      _notifyAdd(item_data.value(),uid);
#endif
    }
    return item_data.value();
  }

  //! Recherche l'entité de numéro unique \a uid
  ItemInternal* findOne(Int64 uid)
  {
#ifdef ARCANE_CHECK
    if (!m_has_unique_id_map)
      _badUniqueIdMap();
#endif
    return m_items_map._tryFindItemInternal(uid);
  }

  //! Vérifie si les structures internes de l'instance sont valides
  void checkValid();

  ItemInternalMap& itemsMap() { return m_items_map; }

  Integer nbItem() const { return m_nb_item; }

  eItemKind kind() const { return m_kind; }

  bool changed() const
  { 
    return !m_added_items.empty() || !m_removed_items.empty();
  }

  void beginCompactItems(ItemFamilyCompactInfos& compact_infos);

  /*!
   * \brief Conversion entre les anciens et les nouveaux id locaux.
   *
   * Cette méthode n'est valide qu'après appel à beginCompactItems() et
   * avant finishCompactItems().
   */
  ARCANE_DEPRECATED_240 Int32ConstArrayView oldToNewLocalIds() const;

  //! Supprime toutes les entités
  void clear();
  
  /*!
   * \brief Conversion entre les nouveaux et les anciens id locaux.
   *
   * Cette méthode n'est valide qu'après appel à beginCompactItems() et
   * avant finishCompactItems().
   */
  ARCANE_DEPRECATED_240 Int32ConstArrayView newToOldLocalIds() const;

  void finishCompactItems(ItemFamilyCompactInfos& compact_infos);

  void itemsUniqueIdToLocalId(ArrayView<Int64> ids,bool do_fatal) const;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              Int64ConstArrayView unique_ids,
                              bool do_fatal) const;
  void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                              ConstArrayView<ItemUniqueId> unique_ids,
                              bool do_fatal) const;

  ItemFamily* itemFamily() const
  {
    return m_item_family;
  }

  void setItemFamily(ItemFamily* item_family);

  bool hasUniqueIdMap() const
  {
    return m_has_unique_id_map;
  }
  
  void setHasUniqueIdMap(bool v);

  void printFreeInternals(Integer max_print);

 public:

  EventObservableView<const ItemFamilyItemListChangedEventArgs&> itemListChangedEvent();

 private:

  /*! \brief Ajoute une entité.
   */
  ItemInternal* _allocOne(bool& need_alloc)
  {
    ItemInternal* new_item = 0;
    Integer nb_free = m_free_internals.size();
    Int32 lid = 0;
    if (nb_free!=0){
      new_item = m_internals[m_free_internals.back()];
      m_free_internals.popBack();
      _setAdded(new_item);
      lid = new_item->localId();
      need_alloc = false;
    }
    else{
      Integer nb_free2 = m_free_internals_in_multi_buffer.size();
      if (nb_free2!=0){
        new_item = m_free_internals_in_multi_buffer.back();
        m_free_internals_in_multi_buffer.popBack();
      }
      else
        new_item = m_item_internals_buffer->allocOne();
      lid = m_internals.size();
      new_item->setLocalId(lid);
      m_internals.add(new_item);
      _updateItemSharedInfoInternalView();
      need_alloc = true;
    }
    m_added_items.add(lid);
    ++m_nb_item;
    return new_item;
  }

 private:

  IMesh* m_mesh; //!< Maillage associé
  ItemFamily* m_item_family; //!< Famille de maillage associée
  eItemKind m_kind; //!< Genre correspondant
  String m_kind_name; //!< Nom du genre des entités (Node, Cell, ...)
  String m_all_group_name; //!< Nom du groupe contenant toutes les entités
  ItemInternalMap m_items_map; //!< Table de hachage conversion uniqueId() -> ItemInternal*
  ItemGroup m_all_group; //! Groupe de toutes les entités
  Integer m_nb_item; //!< Nombre d'entités allouées
  bool m_is_verbose;
  Int32UniqueArray m_added_items;
  Int32UniqueArray m_removed_items;
  bool m_use_new_finalize;
  bool m_is_first_finalize;
  bool m_has_unique_id_map;
  //! Temporaire tant que oldToNewLocalIds() et newToOldLocalIds() existent
  ItemFamilyCompactInfos* m_compact_infos;
  ItemSharedInfo* m_common_item_shared_info = nullptr;
  EventObservable<const ItemFamilyItemListChangedEventArgs&> m_item_list_change_event;

 public:
  
  UniqueArray<ItemInternal*> m_internals; //!< ItemInternal des entités
  Int32UniqueArray m_free_internals; //!< Liste des ItemInternal de m_internals libérés
  //!< Liste des ItemInternal de m_item_internals_buffer libres
  UniqueArray<ItemInternal*> m_free_internals_in_multi_buffer;

 private:

  /*! \brief Tampon pour stocker une instance de ItemInternal.
   *
   * \warning Une instance créée doit rester valide tout au long
   * d'une exécution.
   */
  MultiBufferT<ItemInternal>* m_item_internals_buffer;

 private:

  inline void _setSuppressed(ItemInternal* item)
  {
    int f = item->flags();
    f &= ~ItemFlags::II_Added;
    f |= ItemFlags::II_Suppressed;
    item->setFlags(f);
  }

  inline void _setAdded(ItemInternal* item)
  {
    int f = item->flags();
    f |= ItemFlags::II_Added;
    f &= ~ItemFlags::II_Suppressed;
    item->setFlags(f);
  }

  void _checkActiveItem(ItemInternal* item);
  void _dumpList();
  void _badSameUniqueId(Int64 unique_id) const;
  void _badUniqueIdMap() const;
  void _updateItemSharedInfoInternalView();
  void _notifyRemove(ItemInternal* item)
  {
    if (m_item_list_change_event.hasObservers())
      _notifyRemove2(item);
  }
  void _notifyAdd(ItemInternal* item,Int64 uid)
  {
    if (m_item_list_change_event.hasObservers())
      _notifyAdd2(item,uid);
  }
  void _notifyRemove2(ItemInternal* item);
  void _notifyAdd2(ItemInternal* item,Int64 uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
