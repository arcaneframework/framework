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
class ItemSharedInfoWithType;
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
  friend class mesh::ItemSharedInfoWithType;

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

 public:

  eItemKind itemKind() const { return m_item_kind; }
  IItemFamily* itemFamily() const { return m_item_family; }
  Int32 nbParent() const { return m_nb_parent; }
  ItemTypeInfo* typeInfoFromId(Int32 type_id) const;

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 nbNode() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbEdge() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbFace() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbCell() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbHParent() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 0")
  constexpr Int32 nbHChildren() const { return 0; }

  Int32 typeId() const { return m_type_id; }

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstNode() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstEdge() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstFace() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstCell() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstParent() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstHParent() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  Int32 firstHChild() const { return 0; }

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 neededMemory() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 minimumNeededMemory() const { return 0; }
  ARCCORE_DEPRECATED_2021("This method always return 'false'")
  constexpr bool hasLegacyConnectivity() const { return false; }

 public:

  void print(std::ostream& o) const;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This list is always empty")
  ItemInternalVectorView nodes(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView edges(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView faces(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView cells(Int32) const { return ItemInternalVectorView(); }

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView hChildren(Int32) const { return ItemInternalVectorView(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _parentV2() instead")
  ItemInternal* parent(Integer,Integer) const;

 private:

  ItemInternal* _parentV2(Int32 local_id,[[maybe_unused]] Integer aindex) const
  {
    // Actuellement un seul parent est supporté donc \a aindex doit valoir 0.
    ARCANE_ASSERT((aindex==0),("Only one parent access implemented"));
    return _parents()[(*m_parent_item_ids)[local_id]];
  }
  Int32 _parentLocalIdV2(Int32 local_id,[[maybe_unused]] Integer aindex) const
  {
    // Actuellement un seul parent est supporté donc \a aindex doit valoir 0.
    ARCANE_ASSERT((aindex==0),("Only one parent access implemented"));
    return (*m_parent_item_ids)[local_id];
  }
  void _setParentV2(Int32 local_id,Integer aindex,Int32 parent_local_id) const;
  Int32* _parentPtr(Int32 local_id) const;

 public:

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* node(Int32,Int32) const { return nullptr; }
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

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 nodeLocalId(Int32,Int32) const { return NULL_ITEM_LOCAL_ID; }
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

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setNode(Int32,Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setEdge(Int32,Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setFace(Int32,Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setCell(Int32,Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setHParent(Int32,Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setHChild(Int32,Int32,Int32) const;

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setParentV2() instead")
  void setParent(Integer,Integer,Integer) const;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 edgeAllocated() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 faceAllocated() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 cellAllocated() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 hParentAllocated() const { return 0; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Int32 hChildAllocated() const { return 0; }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always returns 'nullptr'")
  const Int32* _infos() const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void _setInfos(Int32* ptr);

 public:

  MeshItemInternalList* m_items = nullptr;
  ItemInternalConnectivityList* m_connectivity;
  IItemFamily* m_item_family = nullptr;
  ItemTypeMng* m_item_type_mng = nullptr;
  Int64ArrayView* m_unique_ids = nullptr;
  Int32ArrayView* m_parent_item_ids = nullptr;
  Int32ArrayView* m_owners = nullptr;
  Int32ArrayView* m_flags = nullptr;
  eItemKind m_item_kind = IK_Unknown;

 private:

  Int32 m_type_id = IT_NullType;
  Int32 m_nb_parent = 0;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _ownerV2() instead")
  Int32 owner(Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setOwnerV2() instead")
  void setOwner(Int32,Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _flagsV2() instead")
  Int32 flags(Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setFlagsV2() instead")
  void setFlags(Int32,Int32) const;

 private:

  Int32 _ownerV2(Int32 local_id) const { return (*m_owners)[local_id]; }
  void _setOwnerV2(Int32 local_id,Int32 aowner) const { (*m_owners)[local_id] = aowner; }
  Int32 _flagsV2(Int32 local_id) const { return (*m_flags)[local_id]; }
  void _setFlagsV2(Int32 local_id,Int32 f) const { (*m_flags)[local_id] = f; }

 public:

  // TODO: a supprimer
  ARCANE_DEPRECATED_REASON("Y2022: COMMON_BASE_MEMORY is always 0")
  static const Int32 COMMON_BASE_MEMORY = 0;

 private:

  void _init(eItemKind ik);
  //! Version non optimisée mais robuste d'accès à l'ItemInternalArrayView parent
  ItemInternalArrayView _parents() const;
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
