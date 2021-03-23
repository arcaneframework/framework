// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CompactIncrementalItemConnectivity.h                        (C) 2000-2020 */
/*                                                                           */
/* Gestion des connectivités utilisant la méthode compacte.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_COMPACTINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_MESH_COMPACTINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ConnectivityItemVector.h"
#include "arcane/MeshUtils.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe de base de gestion des connectivités utilisant la méthode compacte.
 *
 * La méthode compacte est la méthode historique Arcane qui regroupe
 * toutes les connectivités dans un seul bloc mémoire contigu. Cela permet
 * de réduire l'empreinte mémoire mais fige les connectivités possibles à la
 * compilation.
 *
 * \note Cette classe a besoin d'avoir la vision directe de ItemFamily.
 *
 * Cette classe est abstraite et il doit exister une implémentation spécifique
 * par couple (source_family,target_family).
 *
 * Pour l'instant, les implémentations suivantes sont disponibles:
 * - NodeFaceCompactIncrementalItemConnectivity.
 */
class ARCANE_MESH_EXPORT CompactIncrementalItemConnectivity
: public AbstractIncrementalItemConnectivity
{
 private:
  class Impl;
 public:

  CompactIncrementalItemConnectivity(ItemFamily* source_family,
                                     IItemFamily* target_family,
                                     const String& aname);
  ~CompactIncrementalItemConnectivity();

 public:

  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override
  {
    // Pour l'instant ne fait rien car c'est la famille source qui gère cela directement
    ARCANE_UNUSED(new_to_old_ids);
  }
  void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) override
  {
    // Pour l'instant ne fait rien car c'est la famille cible qui gère cela directement
    ARCANE_UNUSED(old_to_new_ids);
  }  
  void notifySourceItemAdded(ItemLocalId item_local_id) override;
  void notifyReadFromDump() override;

  Integer preAllocatedSize() const override final { return m_pre_allocated_size; }
  void setPreAllocatedSize(Integer value) override final
  {
    m_pre_allocated_size = value;
  }
  void dumpStats(ostream&) const override {}

 protected:

  void _initializeStorage(ConnectivityItemVector* civ) override
  {
    ARCANE_UNUSED(civ);
  }

 protected:

  ItemInternalList m_items_internal;
  ItemFamily::CompactConnectivityHelper m_true_source_family;
  Integer m_pre_allocated_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* TODO: Cette classe n'a rien de spécifique aux connectivités compactes
 * donc on pourrait la rendre accessible ailleurs.
 */
/*!
 * \brief Les classes suivantes permettant de gérer la connectivité historique
 * de ItemInternal pour une connectiivité donnée.
 * Ces classes doivent nécéssairement implémenter les méthodes suivantes:
 \begincode
 * static Integer connectivityIndex();
 * static Integer nbConnectedItem(ItemInternal* item);
 * static Int32 connectedItemLocalId(ItemInternal* item,Integer index);
 * static Int32ArrayView connectedItemsLocalId(ItemInternal* item);
 * static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid);
 * static void updateSharedInfoRemove(ItemFamily::CompactConnectivityHelper& helper,
 *                                    ItemInternal* item,Integer nb_sub_item);
 * static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
 *                                   ItemInternal* item,Integer nb_sub_item);
 \endcode
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux noeuds
class NodeCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::NODE_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbNode();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->nodeLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbNode(),item->_nodesPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setNode(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setNode(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                      ItemInternal* item,Integer nb_sub_item)
  {
    ARCANE_UNUSED(helper);
    ARCANE_UNUSED(item);
    ARCANE_UNUSED(nb_sub_item);
    throw NotSupportedException(A_FUNCINFO);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                    ItemInternal* item,Integer nb_sub_item)
  {
    ARCANE_UNUSED(helper);
    ARCANE_UNUSED(item);
    ARCANE_UNUSED(nb_sub_item);
    throw NotSupportedException(A_FUNCINFO);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux arêtes
class EdgeCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::EDGE_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbEdge();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->edgeLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbEdge(),item->_edgesPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setEdge(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setEdge(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                      ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoRemoved(item,nb_sub_item,0,0);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                    ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoAdded(item,nb_sub_item,0,0);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux faces
class FaceCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::FACE_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbFace();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->faceLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbFace(),item->_facesPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setFace(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setFace(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                     ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoRemoved(item,0,nb_sub_item,0);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                    ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoAdded(item,0,nb_sub_item,0);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux mailles
class CellCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::CELL_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbCell();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->cellLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbCell(),item->_cellsPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setCell(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setCell(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                      ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoRemoved(item,0,0,nb_sub_item);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                    ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoAdded(item,0,0,nb_sub_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux HParent
class HParentCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HPARENT_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbHParent();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->_hParentLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbHParent(),item->_hParentPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setHParent(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setHParent(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                     ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoRemoved(item,0,0,0,nb_sub_item,0);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                    ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoAdded(item,0,0,0,nb_sub_item,0);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux HParent
class HChildCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HCHILD_IDX; }
  static Integer nbConnectedItem(ItemInternal* item)
  {
    return item->nbHChildren();
  }
  static Int32 connectedItemLocalId(ItemInternal* item,Integer index)
  {
    return item->_hChildLocalId(index);
  }
  static Int32ArrayView connectedItemsLocalId(ItemInternal* item)
  {
    return Int32ArrayView(item->nbHChildren(),item->_hChildPtr());
  }
  static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid)
  {
    item->_setHChild(index,target_lid);
  }
  static void replaceConnectedItems(ItemInternal* item,Int32ConstArrayView target_lids)
  {
    for( Integer i=0, n=target_lids.size(); i<n; ++i )
      item->_setHChild(i,target_lids[i]);
  }
  static void updateSharedInfoRemoved(ItemFamily::CompactConnectivityHelper& helper,
                                     ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoRemoved(item,0,0,0,0,nb_sub_item);
  }
  static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
                                     ItemInternal* item,Integer nb_sub_item)
  {
    helper.updateSharedInfoAdded(item,0,0,0,0,nb_sub_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de la connectivité compacte dont la famille cible
 * utilise l'accesseur \a AccessorType
 */
template<typename AccessorType>
class CompactIncrementalItemConnectivityT
: public CompactIncrementalItemConnectivity
{
 public:
  static Integer connectivityIndex() { return AccessorType::connectivityIndex(); }
 public:
  CompactIncrementalItemConnectivityT(ItemFamily* source_family,
                                      IItemFamily* target_family,
                                      const String& aname)
  : CompactIncrementalItemConnectivity(source_family,target_family,aname){}

 public:

  Integer nbConnectedItem(ItemLocalId lid) const override
  {
    return AccessorType::nbConnectedItem(m_items_internal[lid]);
  }
  Int32 connectedItemLocalId(ItemLocalId lid,Integer index) const override
  {
    return AccessorType::connectedItemLocalId(m_items_internal[lid],index);
  }
  ItemVectorView _connectedItems(ItemLocalId lid,ConnectivityItemVector& con_items) const override
  {
    ItemInternal* item = m_items_internal[lid];
    return con_items.resizeAndCopy(AccessorType::connectedItemsLocalId(item));
  }
  void replaceConnectedItem(ItemLocalId source_lid,Integer index,ItemLocalId target_lid) final
  {
    ItemInternal* source_item = m_true_source_family.itemInternal(source_lid);
    AccessorType::replaceConnectedItem(source_item,index,target_lid);
  }
  void replaceConnectedItems(ItemLocalId source_lid,Int32ConstArrayView target_lids) final
  {
    ItemInternal* source_item = m_true_source_family.itemInternal(source_lid);
    AccessorType::replaceConnectedItems(source_item,target_lids);
  }
  void addConnectedItem(ItemLocalId source_lid,ItemLocalId new_lid) override
  {
    // NOTE: cette méhode est invalide pour ItemType==Node
    ItemInternal* item = m_true_source_family.itemInternal(source_lid);
    Integer nb_sub_item = AccessorType::nbConnectedItem(item);
    AccessorType::updateSharedInfoAdded(m_true_source_family,item,1);
    AccessorType::replaceConnectedItem(item,nb_sub_item,new_lid);
  }
  void removeConnectedItem(ItemLocalId source_lid,ItemLocalId remove_lid) override
  {
    // NOTE: cette méhode est invalide pour ItemType==Node
    ItemInternal* item = m_true_source_family.itemInternal(source_lid);
    Integer nb_sub_item = AccessorType::nbConnectedItem(item);
    if (nb_sub_item==0)
      ARCANE_FATAL("Can not remove sub_item lid={0} from item uid={1} without sub_item",
                   remove_lid, item->uniqueId());
    mesh_utils::removeItemAndKeepOrder(AccessorType::connectedItemsLocalId(item),remove_lid);
    AccessorType::updateSharedInfoRemoved(m_true_source_family,item,1);
  }
  void removeConnectedItems(ItemLocalId source_lid) override
  {
    // NOTE: cette méhode est invalide pour ItemType==Node
    ItemInternal* item = m_true_source_family.itemInternal(source_lid);
    Integer nb_sub_item = AccessorType::nbConnectedItem(item);
    if (nb_sub_item!=0)
      AccessorType::updateSharedInfoRemoved(m_true_source_family,item,nb_sub_item);
  }

  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override
  {
    bool has_connection = false;
    for (Integer i = 0; i < nbConnectedItem(source_item);++i) {
      if (connectedItemLocalId(source_item, i) == target_local_id.localId())
        has_connection = true;
    }
    return has_connection;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
