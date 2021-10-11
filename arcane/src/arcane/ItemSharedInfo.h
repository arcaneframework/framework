// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.h                                            (C) 2000-2020 */
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
 public:

  friend class ItemInternal;
  static const Int32 NULL_INDEX = static_cast<Int32>(-1);

 public:
  //! Pour l'entité nulle
  static ItemSharedInfo nullItemSharedInfo;
 public:
  ItemSharedInfo();
  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids);
  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
                 Int32 nb_edge,Int32 nb_face,Int32 nb_cell);
  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
                 Int32 nb_edge,Int32 nb_face,Int32 nb_cell,
                 Int32 edge_allocated,Int32 face_allocated,Int32 cell_allocated);
  //! AMR
  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
                 Int32 nb_edge,Int32 nb_face,Int32 nb_cell,
                 Int32 nb_parent,Int32 nb_children,
                 Int32 edge_allocated,Int32 face_allocated,Int32 cell_allocated,
                 Int32 parent_allocated,Int32 child_allocated);

  ItemSharedInfo(IItemFamily* family,ItemTypeInfo* item_type,MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity,Int64ArrayView* unique_ids,
                 Int32ConstArrayView buffer);
 public:
  eItemKind itemKind() const { return m_item_kind; }
  IItemFamily* itemFamily() const { return m_item_family; }
  Int32 nbNode() const { return m_nb_node; }
  Int32 nbEdge() const { return m_nb_edge; }
  Int32 nbFace() const { return m_nb_face; }
  Int32 nbCell() const { return m_nb_cell; }
  Int32 nbParent() const { return m_nb_parent; }
  //! AMR
  Int32 nbHParent() const { return m_nb_hParent; }
  Int32 nbHChildren() const {return m_nb_hChildren; }

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
  ItemInternalVectorView edges(Int32 data_index) const
  { return ItemInternalVectorView(m_items->edges,m_infos+data_index+firstEdge(),nbEdge()); }
  ItemInternalVectorView faces(Int32 data_index) const
  { return ItemInternalVectorView(m_items->faces,m_infos+data_index+firstFace(),nbFace()); }
  ItemInternalVectorView cells(Int32 data_index) const
  { return ItemInternalVectorView(m_items->cells,m_infos+data_index+firstCell(),nbCell()); }

  //!AMR
  ItemInternalVectorView hChildren(Int32 data_index) const
  { return ItemInternalVectorView(m_items->cells,m_infos+data_index+firstHChild(),nbHChildren()); }
 public:
  ItemInternal* node(Int32 data_index,Int32 aindex) const
  { return m_items->nodes[ m_infos[data_index + firstNode()+aindex] ]; }
  ItemInternal* edge(Int32 data_index,Int32 aindex) const
  { return m_items->edges[ m_infos[data_index + firstEdge()+aindex] ]; }
  ItemInternal* face(Int32 data_index,Int32 aindex) const
  { return m_items->faces[ m_infos[data_index + firstFace()+aindex] ]; }
  ItemInternal* cell(Int32 data_index,Int32 aindex) const
  { return m_items->cells[ m_infos[data_index + firstCell()+aindex] ]; }
  //! AMR
  ItemInternal* parent(Integer data_index,Integer aindex) const
  { return _parents(aindex)[m_infos[data_index + firstParent()+aindex] ]; }
  ItemInternal* hParent(Integer data_index,Integer aindex) const
  { return m_items->cells[m_infos[data_index + firstHParent()+aindex] ]; }
  ItemInternal* hChild(Int32 data_index,Int32 aindex) const
  { return m_items->cells[m_infos[data_index + firstHChild()+aindex] ]; }

 public:
  Int32 nodeLocalId(Int32 data_index,Int32 aindex) const
  { return m_infos[data_index+firstNode()+aindex]; }
  Int32 edgeLocalId(Int32 data_index,Int32 aindex) const
  { return m_infos[data_index+firstEdge()+aindex]; }
  Int32 faceLocalId(Int32 data_index,Int32 aindex) const
  { return m_infos[data_index+firstFace()+aindex]; }
  Int32 cellLocalId(Int32 data_index,Int32 aindex) const
  { return m_infos[data_index+firstCell()+aindex]; }
  Integer parentLocalId(Integer data_index,Integer aindex) const
  { return m_infos[data_index+firstParent()+aindex]; }
  //! AMR
  Int32 hParentLocalId(Integer data_index,Integer aindex) const
  { return m_infos[data_index+firstHParent()+aindex]; }
  Int32 hChildLocalId(Integer data_index,Integer aindex) const
  { return m_infos[data_index+firstHChild()+aindex]; }

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
  Int32 edgeAllocated() const { return m_edge_allocated; }
  Int32 faceAllocated() const { return m_face_allocated; }
  Int32 cellAllocated() const { return m_cell_allocated; }
  //!AMR
  Int32 hParentAllocated() const { return m_hParent_allocated; }
  Int32 hChildAllocated() const { return m_hChild_allocated; }

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
  // TODO: GG: ne devrait pas être statique
  static bool m_is_amr_activated;

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
