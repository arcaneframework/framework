// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.h                                            (C) 2000-2022 */
/*                                                                           */
/* Informations communes à plusieurs entités.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMSHAREDINFO_H
#define ARCANE_ITEMSHAREDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/ItemTypeInfo.h"
#include "arcane/ItemInternalVectorView.h"
#include "arcane/MeshItemInternalList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemSharedInfoList;
class ItemFamily;
}

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class ItemInternalConnectivityList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Structure interne partagée d'une entité de maillage.

 Cette classe renferme des informations communes à plusieurs entités.
 Il s'agit d'une implémentation du design pattern FlyWeight. Cela
 permet de faire des économies mémoire importantes lorsque le maillage
 comporte un grand nombre d'entités similaire et qu'il est en grande
 partie structuré.

 Comme une instance de cette classe est partagée par plusieurs entités, il
 ne faut en cas la modifier directement. C'est à l'implémentation (Mesh)
 de fournir un mécanisme gérant les instances de cette classe.

 Parmi les informations partagées se trouvent le nombre des entités
 connectées à cette entité (nombre de noeuds, d'arêtes, de faces et
 de mailles) mais aussi les informations locales à un type d'entité
 donné comme par exemple la liste et la connectivité locale des
 faces d'un héxaèdre. Ces informations sont directements gérées par
 la classe ItemTypeInfo.

 \todo Il ne faut plus utiliser les localFace(), localEdge() mais
 passer par l'intermédiaire de m_item_type.
 */
class ARCANE_CORE_EXPORT ItemSharedInfo
{
 private:

  /*!
   * Liste des vues sur les variabes associées aux entités.
   *
   * Ces variables sont toutes indexables par le localId()
   * de l'entité. Elles sont toujours allouées sauf 'm_parent_ids_view'
   * qui n'est alloué que si l'entité est dans un sous-maillage
   */
  struct ItemVariableViews
  {
    Int64ArrayView m_unique_ids_view;
    Int32ArrayView m_flags_view;
    Int32ArrayView m_owners_view;
    Int32ArrayView m_parent_ids_view;
  };

 public:

  friend class ItemInternal;
  friend class mesh::ItemSharedInfoList;
  friend class mesh::ItemFamily;

  static const Int32 NULL_INDEX = static_cast<Int32>(-1);

 public:

  //! Pour l'entité nulle
  static ItemSharedInfo nullItemSharedInfo;

 public:

  ItemSharedInfo();

 private:

  // Seule ItemSharedInfoList peut créer des instances de cette classe autre que
  // l'instance nulle.
  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,ItemVariableViews* variable_views);

  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,ItemVariableViews* variable_views,
                 Int32ConstArrayView buffer);
 public:

  eItemKind itemKind() const { return m_item_kind; }
  IItemFamily* itemFamily() const { return m_item_family; }
  Int32 nbParent() const { return m_nb_parent; }
  Int32 nbNode() const { return m_nb_node; }

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbEdge() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  Int32 nbFace() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  Int32 nbCell() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  Int32 nbHParent() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  Int32 nbHChildren() const { return 0; }

  Int32 typeId() const { return m_type_id; }

  Int32 firstNode() const { return m_first_node; }
  Int32 firstEdge() const { return m_first_edge; }
  Int32 firstFace() const { return m_first_face; }
  Int32 firstCell() const { return m_first_cell; }
  Int32 firstParent() const { return m_first_parent; }
  //! AMR
  Int32 firstHParent() const { return m_first_hParent; }
  Int32 firstHChild() const {return m_first_hChild; }

  Int32 neededMemory() const { return m_needed_memory; }
  Int32 minimumNeededMemory() const { return m_minimum_needed_memory; }
  ARCCORE_DEPRECATED_2021("This method always return 'false'")
  constexpr bool hasLegacyConnectivity() const { return false; }

 public:

  void print(std::ostream& o) const;

 public:

  ItemInternalVectorView nodes(Int32 data_index) const
  { return ItemInternalVectorView(m_items->nodes,m_infos+data_index+firstNode(),nbNode()); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView edges(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView faces(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView cells(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView hChildren(Int32) const { return ItemInternalVectorView(); }

 public:

  ItemInternal* parent(Integer data_index,Integer aindex) const
  { return _parents(aindex)[m_infos[data_index + firstParent()+aindex] ]; }

  ItemInternal* node(Int32 data_index,Int32 aindex) const
  { return m_items->nodes[ m_infos[data_index + firstNode()+aindex] ]; }

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* edge(Int32,Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* face(Int32,Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* cell(Int32,Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* hParent(Integer,Integer) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* hChild(Int32,Int32) const { return nullptr; }

 public:

  Int32 nodeLocalId(Int32 data_index,Int32 aindex) const
  { return m_infos[data_index+firstNode()+aindex]; }

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 edgeLocalId(Int32,Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 faceLocalId(Int32,Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 cellLocalId(Int32,Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Integer parentLocalId(Integer,Integer) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 hParentLocalId(Integer,Integer) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 hChildLocalId(Integer,Integer) const { return NULL_ITEM_LOCAL_ID; }

 public:

  void setNode(Int32 data_index,Int32 aindex,Int32 local_id) const
  { m_infos[ data_index+firstNode()+aindex] = local_id; }
  void setEdge(Int32 data_index,Int32 aindex,Int32 local_id) const
  { m_infos[ data_index+firstEdge()+aindex] = local_id; }
  void setFace(Int32 data_index,Int32 aindex,Int32 local_id) const
  { m_infos[ data_index+firstFace()+aindex] = local_id; }
  void setCell(Int32 data_index,Int32 aindex,Int32 local_id) const
  { m_infos[ data_index+firstCell()+aindex] = local_id; }
  void setParent(Integer data_index,Integer aindex,Integer local_id) const
  { m_infos[ data_index+firstParent()+aindex] = local_id; }
  //! AMR
  void setHParent(Integer data_index,Integer aindex,Integer local_id) const
  { m_infos[ data_index+firstHParent()+aindex] = local_id; }
  void setHChild(Int32 data_index,Int32 aindex,Int32 local_id) const
  { m_infos[ data_index+firstHChild()+aindex] = local_id; }

 public:

  constexpr Int32 edgeAllocated() const { return 0; }
  constexpr Int32 faceAllocated() const { return 0; }
  constexpr Int32 cellAllocated() const { return 0; }
  constexpr Int32 hParentAllocated() const { return 0; }
  constexpr Int32 hChildAllocated() const { return 0; }

 public:
  const Int32* _infos() const { return m_infos; }
  void _setInfos(Int32* ptr) { m_infos = ptr; }
 private:
  Int32* m_infos = nullptr;
 private:
  Int32 m_first_node = 0;
  Int32 m_nb_node = 0;
  Int32 m_first_edge = 0;
  Int32 m_nb_edge = 0;
  Int32 m_first_face = 0;
  Int32 m_nb_face = 0;
  Int32 m_first_cell = 0;
  Int32 m_nb_cell = 0;
  Int32 m_first_parent = 0;
  Int32 m_nb_parent = 0;
  //! AMR
  Int32 m_first_hParent = 0;
  Int32 m_first_hChild = 0;
  Int32 m_nb_hParent = 0;
  Int32 m_nb_hChildren = 0;

 public:
  MeshItemInternalList* m_items = nullptr;
  ItemInternalConnectivityList* m_connectivity;
  IItemFamily* m_item_family = nullptr;
  Int64ArrayView* m_unique_ids = nullptr;
  ItemTypeInfo* m_item_type = nullptr;
  eItemKind m_item_kind = IK_Unknown;
 private:
  Int32 m_needed_memory = 0;
  Int32 m_minimum_needed_memory = 0;
  Int32 m_edge_allocated = 0;
  Int32 m_face_allocated = 0;
  Int32 m_cell_allocated = 0;
  //! AMR
  Int32 m_hParent_allocated = 0;
  Int32 m_hChild_allocated = 0;

  Int32 m_type_id = IT_NullType;
  Int32 m_index = NULL_INDEX;
  Int32 m_nb_reference = 0;
 public:
  Int32 index() const { return m_index; }
  void setIndex(Int32 aindex) { m_index = aindex; }
  Int32 nbReference() const { return m_nb_reference; }
  void addReference(){ ++m_nb_reference; }
  void removeReference(){ --m_nb_reference; }
  void serializeWrite(Int32ArrayView buffer);
  static Integer serializeWriteSize();
  static Integer serializeSize();
  static Integer serializeAMRSize();
  static Integer serializeNoAMRSize();
 public:

  Int32 owner(Int32 data_index) const
  { return m_infos[data_index+OWNER_INDEX]; }
  void setOwner(Int32 data_index,Int32 aowner) const
  { m_infos[data_index+OWNER_INDEX] = aowner; }

 public:

  Int32 flags(Int32 data_index) const
  { return m_infos[data_index+FLAGS_INDEX]; }
  void setFlags(Int32 data_index,Int32 f) const
  { m_infos[data_index+FLAGS_INDEX] = f; }

 private:

  static const Int32 OWNER_INDEX = 0;
  static const Int32 FLAGS_INDEX = 1;
  static const Int32 FIRST_NODE_INDEX = 2;

 public:

  static const Int32 COMMON_BASE_MEMORY = 2;

 private:

  void _init(eItemKind ik);
  //! Version non optimisé mais robuste d'accès à l'ItemInternalArrayView parent
  ItemInternalArrayView _parents(Integer index) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const ItemSharedInfo& isi)
{
  isi.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
