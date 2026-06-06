// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfo.h                                            (C) 2000-2025 */
/*                                                                           */
/* Common information for multiple entities.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMSHAREDINFO_H
#define ARCANE_CORE_ITEMSHAREDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemTypeInfo.h"
#include "arcane/core/MeshItemInternalList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemFamily;
class ItemSharedInfoWithType;
class DynamicMeshKindInfos;
} // namespace Arcane::mesh

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class ItemInternalVectorView;
class ItemInternalConnectivityList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal shared structure of a mesh entity.
 *
 * This class holds common information for multiple entities.
 *
 * Since an instance of this class is shared by multiple entities, it
 * should not be modified directly. It is up to the implementation (Mesh)
 * to provide a mechanism managing instances of this class.
 */
class ARCANE_CORE_EXPORT ItemSharedInfo
{
  friend class ItemBase;
  friend class MutableItemBase;
  friend class Item;
  friend class ItemGenericInfoListView;
  friend class ItemInternal;
  friend class ItemInfoListView;
  friend class mesh::ItemFamily;
  friend class mesh::DynamicMeshKindInfos;
  friend class mesh::ItemSharedInfoWithType;
  friend class ItemInternalVectorView;
  friend class ItemVectorViewConstIterator;
  friend class ItemConnectedListViewConstIterator;
  friend class ItemVectorView;
  friend class ItemEnumeratorBase;
  friend class ItemInternalCompatibility;
  friend class SimdItemBase;
  friend class SimdItemEnumeratorBase;
  template <int Extent> friend class ItemConnectedListView;

 public:

  static const Int32 NULL_INDEX = static_cast<Int32>(-1);

 public:

  // TODO: Make private. Access must now go through nullInstance()
  //! For the null entity
  static ItemSharedInfo nullItemSharedInfo;

 private:

  static ItemSharedInfo* nullItemSharedInfoPointer;

 public:

  static ItemSharedInfo* nullInstance() { return nullItemSharedInfoPointer; }

 public:

  ItemSharedInfo();

 private:

  // Only ItemFamily can create instances of this class other than the null instance.
  ItemSharedInfo(IItemFamily* family, MeshItemInternalList* items,
                 ItemInternalConnectivityList* connectivity);

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

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use ItemInternal::typeId() instead")
  Int32 typeId() const;

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

  void updateMeshItemInternalList();

 public:

  void print(std::ostream& o) const;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This list is always empty")
  ItemInternalVectorView nodes(Int32) const;

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView edges(Int32) const;

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView faces(Int32) const;

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView cells(Int32) const;

  ARCANE_DEPRECATED_REASON("Y2020: This list is always empty")
  ItemInternalVectorView hChildren(Int32) const;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _parentV2() instead")
  ItemInternal* parent(Integer, Integer) const;

 private:

  ItemInternal* _parentV2(Int32 local_id, [[maybe_unused]] Integer aindex) const
  {
    // Currently only one parent is supported, so \a aindex must be 0.
    ARCANE_ASSERT((aindex == 0), ("Only one parent access implemented"));
    return _parent(m_parent_item_ids[local_id]);
  }
  Int32 _parentLocalIdV2(Int32 local_id, [[maybe_unused]] Integer aindex) const
  {
    // Currently only one parent is supported, so \a aindex must be 0.
    ARCANE_ASSERT((aindex == 0), ("Only one parent access implemented"));
    return m_parent_item_ids[local_id];
  }
  void _setParentV2(Int32 local_id, Integer aindex, Int32 parent_local_id);
  Int32* _parentPtr(Int32 local_id);

 public:

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* node(Int32, Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* edge(Int32, Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* face(Int32, Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* cell(Int32, Int32) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* hParent(Integer, Integer) const { return nullptr; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'nullptr'")
  ItemInternal* hChild(Int32, Int32) const { return nullptr; }

 public:

  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 nodeLocalId(Int32, Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 edgeLocalId(Int32, Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 faceLocalId(Int32, Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 cellLocalId(Int32, Int32) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Integer parentLocalId(Integer, Integer) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 hParentLocalId(Integer, Integer) const { return NULL_ITEM_LOCAL_ID; }
  ARCANE_DEPRECATED_REASON("Y2020: This method always return 'NULL_ITEM_LOCAL_ID'")
  Int32 hChildLocalId(Integer, Integer) const { return NULL_ITEM_LOCAL_ID; }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setNode(Int32, Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setEdge(Int32, Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setFace(Int32, Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setCell(Int32, Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setHParent(Int32, Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setHChild(Int32, Int32, Int32) const;

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setParentV2() instead")
  void setParent(Integer, Integer, Integer) const;

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

 private:

  // ATTENTION:
  // Any modification to the list or the type of fields must be
  // reported in the C# wrapper (tools/wrapper/core/csharp) and
  // in the proxy for totalview (src/arcane/totalview)
  MeshItemInternalList* m_items = nullptr;
  ItemInternalConnectivityList* m_connectivity;
  IItemFamily* m_item_family = nullptr;
  ItemTypeMng* m_item_type_mng = nullptr;
  Int64ArrayView m_unique_ids;
  Int32ArrayView m_parent_item_ids;
  Int32ArrayView m_owners;
  Int32ArrayView m_flags;
  Int16ArrayView m_type_ids;
  eItemKind m_item_kind = IK_Unknown;
  Int32 m_nb_parent = 0;
  ConstArrayView<ItemInternal*> m_items_internal; //!< ItemInternal of entities

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _ownerV2() instead")
  Int32 owner(Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setOwnerV2() instead")
  void setOwner(Int32, Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _flagsV2() instead")
  Int32 flags(Int32) const;
  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception. Use _setFlagsV2() instead")
  void setFlags(Int32, Int32) const;

 private:

  constexpr Int32 _ownerV2(Int32 local_id) const { return m_owners[local_id]; }
  void _setOwnerV2(Int32 local_id, Int32 aowner) { m_owners[local_id] = aowner; }
  constexpr Int32 _flagsV2(Int32 local_id) const { return m_flags[local_id]; }
  void _setFlagsV2(Int32 local_id, Int32 f) { m_flags[local_id] = f; }

  constexpr Int16 _typeId(Int32 local_id) const { return m_type_ids[local_id]; }
  void _setTypeId(Int32 local_id, Int16 v) { m_type_ids[local_id] = v; }

 public:

  // TODO: to be removed
  ARCANE_DEPRECATED_REASON("Y2022: COMMON_BASE_MEMORY is always 0")
  static const Int32 COMMON_BASE_MEMORY = 0;

 private:

  void _init(eItemKind ik);
  //! Non-optimized but robust version of accessing the parent ItemInternalArrayView
  ItemInternal* _parent(Int32 id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o, const ItemSharedInfo& isi)
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
