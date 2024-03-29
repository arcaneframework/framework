﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfoList.h                                        (C) 2000-2022 */
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
 * \brief Classe temporaire pour conserver un ItemSharedInfo et un type d'entité.
 *
 * Cette classe permet de faire la transition entre la version de Arcane où
 * le ItemSharedInfo contient aussi le type de l'entité et les futures versions
 * où cela ne sera pas le cas.
 */
class ItemSharedInfoWithType
{
  friend class ItemSharedInfoList;
  static const Int32 NULL_INDEX = ItemSharedInfo::NULL_INDEX;

 public:

  ItemSharedInfoWithType() : m_shared_info(&ItemSharedInfo::nullItemSharedInfo) {}

 private:

  ItemSharedInfoWithType(ItemSharedInfo* shared_info,ItemTypeInfo* item_type);
  ItemSharedInfoWithType(ItemSharedInfo* shared_info,ItemTypeInfo* item_type,Int32ConstArrayView buffer);

 public:

  ItemSharedInfo* sharedInfo() { return m_shared_info; }
  ItemTypeId itemTypeId() { return this->m_type_id; }
  Int32 index() const { return m_index; }
  void setIndex(Int32 aindex) { m_index = aindex; }
  Int32 nbReference() const { return m_nb_reference; }
  void addReference(){ ++m_nb_reference; }
  void removeReference(){ --m_nb_reference; }
  void serializeWrite(Int32ArrayView buffer);
  static Integer serializeSize() { return 6; }

 public:

  friend std::ostream& operator<<(std::ostream& o,const ItemSharedInfoWithType& isi)
  {
    isi.m_shared_info->print(o);
    return o;
  }

 private:

  ItemSharedInfo* m_shared_info = nullptr;
  ItemTypeId m_type_id;
  Int32 m_index = NULL_INDEX;
  Int32 m_nb_reference = 0;
};

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
  typedef std::map<ItemNumElements,ItemSharedInfoWithType*> ItemSharedInfoMap;

 public:

  ItemSharedInfoList(ItemFamily* family,ItemSharedInfo* common_shared_info);
  //! Libère les ressources
  ~ItemSharedInfoList();

 public:

  ConstArrayView<ItemSharedInfoWithType*> itemSharedInfos() const { return m_item_shared_infos; }
  ArrayView<ItemSharedInfoWithType*> itemSharedInfos() { return m_item_shared_infos; }

 private:

  ItemSharedInfoWithType* allocOne()
  {
    bool need_alloc = false;
    ItemSharedInfoWithType* next = _allocOne(need_alloc);
    return next;
  }

  ItemSharedInfoWithType* allocOne(bool& need_alloc)
  {
    ItemSharedInfoWithType* next = _allocOne(need_alloc);
    return next;
  }

  void removeOne(ItemSharedInfoWithType* item)
  {
    m_list_changed = true;
    m_free_item_shared_infos.add(item);
    --m_nb_item_shared_info;
  }

 public:

  //! Vérifie si les structures internes de l'instance sont valides
  void checkValid();

  Integer nbItemSharedInfo() const { return m_nb_item_shared_info; }

  void prepareForDump();
  void readFromDump();

  void dumpSharedInfos();

  //! Indique si la liste a changée depuis le dernier appel à prepareForDump()
  bool hasChanged() { return m_list_changed; }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxNodePerItem() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxEdgePerItem() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxFacePerItem() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxCellPerItem() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxLocalNodePerItemType() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxLocalEdgePerItemType() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Integer maxLocalFacePerItemType() const { return 0; }

 public:

  ItemSharedInfoWithType* findSharedInfo(ItemTypeInfo* type);

 private:

  ItemSharedInfoWithType* _allocOne(bool& need_alloc)
  {
    ItemSharedInfoWithType* new_item = 0;
    Integer nb_free = m_free_item_shared_infos.size();
    m_list_changed = true;
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

  ItemFamily* m_family = nullptr;
  ItemSharedInfo* m_common_item_shared_info = nullptr;
  Integer m_nb_item_shared_info = 0; //!< Nombre d'objets alloués
  eItemKind m_item_kind = IK_Unknown;
  UniqueArray<ItemSharedInfoWithType*> m_item_shared_infos;
  UniqueArray<ItemSharedInfoWithType*> m_free_item_shared_infos;
  MultiBufferT<ItemSharedInfoWithType>* m_item_shared_infos_buffer;
  ItemSharedInfoMap* m_infos_map = nullptr;
  Variables* m_variables = nullptr;
  bool m_list_changed = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
