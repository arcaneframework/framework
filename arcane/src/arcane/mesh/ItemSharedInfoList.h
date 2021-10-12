// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfoList.h                                        (C) 2000-2020 */
/*                                                                           */
/* Liste de 'ItemSharedInfo'.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMSHAREDINFOLIST_H
#define ARCANE_MESH_ITEMSHAREDINFOLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"

#include "arcane/ItemGroup.h"
#include "arcane/ItemInternal.h"
#include "arcane/VariableTypedef.h"

#include "arcane/mesh/MeshGlobal.h"

#include <map>

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
 * \brief Infos de maillage pour un genre donné d'entité.

 Une instance de cette classe gère toutes les structures de maillage
 pour une entité d'un genre donné.
*/
class ItemSharedInfoList
: public TraceAccessor
{
 public:

  class ItemNumElements;
  class Variables;
  typedef std::map<ItemNumElements,ItemSharedInfo*> ItemSharedInfoMap;

 private:


 public:

  ItemSharedInfoList(ItemFamily* family);
  //! Libère les ressources
  ~ItemSharedInfoList();

 public:

  ConstArrayView<ItemSharedInfo*> itemSharedInfos() const { return m_item_shared_infos; }
  ArrayView<ItemSharedInfo*> itemSharedInfos() { return m_item_shared_infos; }

  ItemSharedInfo* allocOne()
  {
    bool need_alloc = false;
    ItemSharedInfo* next = _allocOne(need_alloc);
    return next;
  }

  ItemSharedInfo* allocOne(bool& need_alloc)
  {
    ItemSharedInfo* next = _allocOne(need_alloc);
    return next;
  }

  void removeOne(ItemSharedInfo* item)
  {
    m_list_changed = true;
    m_connectivity_info_changed = true;
    m_free_item_shared_infos.add(item);
    --m_nb_item_shared_info;
  }

  //! Vérifie si les structures internes de l'instance sont valides
  void checkValid();

  ISubDomain* subDomain();

  Integer nbItemSharedInfo() const { return m_nb_item_shared_info; }

  void prepareForDump();
  void readFromDump();

  void dumpSharedInfos();

  //! Indique si la liste a changée depuis le dernier appel à prepareForDump()
  bool hasChanged() { return m_list_changed; }

 public:
  Integer maxNodePerItem();
  Integer maxEdgePerItem();
  Integer maxFacePerItem();
  Integer maxCellPerItem();
  Integer maxLocalNodePerItemType();
  Integer maxLocalEdgePerItemType();
  Integer maxLocalFacePerItemType();
 public:
  ItemSharedInfo* findSharedInfo4(ItemTypeInfo* type,Integer nb_edge,Integer nb_face,Integer nb_cell)
  {
    return findSharedInfo7(type,nb_edge,nb_face,nb_cell,nb_edge,nb_face,nb_cell);
  }
  ItemSharedInfo* findSharedInfo7(ItemTypeInfo* type,Integer nb_edge,Integer nb_face,
                                 Integer nb_cell,Integer edge_allocated,
                                 Integer face_allocated,Integer cell_allocated);
  //! AMR
  ItemSharedInfo* findSharedInfo6(ItemTypeInfo* type,Integer nb_edge,Integer nb_face,Integer nb_cell,
                                 Integer nb_hParent,Integer nb_hChildren)
  {
    return findSharedInfo11(type,nb_edge,nb_face,nb_cell,nb_hParent,nb_hChildren,
                          nb_edge,nb_face,nb_cell,nb_hParent,nb_hChildren);
  }
  ItemSharedInfo* findSharedInfo11(ItemTypeInfo* type,Integer nb_edge,Integer nb_face,Integer nb_cell,
                                 Integer nb_hParent,Integer nb_hChildren,
                                 Integer edge_allocated,Integer face_allocated,Integer cell_allocated,
                                 Integer hParent_allocated, Integer hChild_allocated);

 public:

  void setSharedInfosPtr(Int32* ptr);

 private:

  ItemSharedInfo* _allocOne(bool& need_alloc)
  {
    ItemSharedInfo* new_item = 0;
    Integer nb_free = m_free_item_shared_infos.size();
    m_list_changed = true;
    m_connectivity_info_changed = true;
    if (nb_free!=0){
      new_item = m_free_item_shared_infos.back();
      m_free_item_shared_infos.popBack();
      need_alloc = false;
    }
    else{
      new_item = m_item_shared_infos_buffer->allocOne();
      new_item->setIndex(m_item_shared_infos.size());
      m_item_shared_infos.add(new_item);
      need_alloc = true;
    }
    ++m_nb_item_shared_info;
    return new_item;
  }

 private:

  ItemFamily* m_family;
  ISubDomain* m_sub_domain; //!< Sous-domaine associé
  Integer m_nb_item_shared_info; //!< Nombre d'objets alloués
  eItemKind m_item_kind;
  UniqueArray<ItemSharedInfo*> m_item_shared_infos;
  UniqueArray<ItemSharedInfo*> m_free_item_shared_infos;
  MultiBufferT<ItemSharedInfo>* m_item_shared_infos_buffer;
  ItemSharedInfoMap* m_infos_map;
  Variables* m_variables;
  bool m_list_changed;
  bool m_connectivity_info_changed;
  Integer m_max_node_per_item;
  Integer m_max_edge_per_item;
  Integer m_max_face_per_item;
  Integer m_max_cell_per_item;
  Integer m_max_node_per_item_type;
  Integer m_max_edge_per_item_type;
  Integer m_max_face_per_item_type;

 private:

  void _checkConnectivityInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
