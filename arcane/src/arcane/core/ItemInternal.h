// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.h                                              (C) 2000-2026 */
/*                                                                           */
/* Internal part of an entity.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINTERNAL_H
#define ARCANE_CORE_ITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemIndexedListView.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemUniqueId.h"
#include "arcane/core/ItemLocalIdListView.h"
#include "arcane/core/ItemTypeId.h"
#include "arcane/core/ItemFlags.h"
#include "arcane/core/ItemConnectivityContainerView.h"
#include "arcane/core/ItemInternalVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef null
#undef null
#endif

//#define ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO

#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
#define A_INTERNAL_SI(name) m_shared_infos.m_##name
#else
#define A_INTERNAL_SI(name) m_items->m_##name##_shared_info
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class IncrementalItemConnectivityBase;
class PolyhedralFamily;
class PolyhedralMeshImpl;
class FaceFamily;
class MeshRefinement;
} // namespace Arcane::mesh
namespace Arcane::Materials
{
class ConstituentItemSharedInfo;
}
namespace Arcane
{
class ItemInternalCompatibility;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class for building an instance of ItemBase
 */
class ARCANE_CORE_EXPORT ItemBaseBuildInfo
{
 public:

  ItemBaseBuildInfo() = default;
  constexpr ItemBaseBuildInfo(Int32 local_id, ItemSharedInfo* shared_info)
  : m_local_id(local_id)
  , m_shared_info(shared_info)
  {}

 public:

  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
  ItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Connectivity information, for an entity family,
 * allowing transition between old and new connectivity implementations.
 */
class ARCANE_CORE_EXPORT ItemInternalConnectivityList
{
  // IMPORTANT: This structure must have the same memory layout
  // as the C# structure of the same name.

  friend class ItemBase;
  friend class ItemInternal;
  friend class Item;

  // For access to _setConnectivity*
  friend mesh::IncrementalItemConnectivityBase;
  friend mesh::PolyhedralFamily;
  friend mesh::PolyhedralMeshImpl;

  // For access to m_items
  friend mesh::ItemFamily;

 private:

  /*!
   * \brief Specific view to manage null entities.
   *
   * For the null entity, the index is NULL_ITEM_LOCAL_ID (i.e., -1) and it must
   * be possible to access `m_data` with this index, which is not possible
   * with the classic ArrayView in check mode.
   */
  struct Int32View
  {
   public:

    constexpr Int32 operator[](Int32 index) const
    {
#ifdef ARCANE_CHECK
      if (index == NULL_ITEM_LOCAL_ID) {
        // For the null entity, the size must be 0.
        if (m_size != 0)
          arcaneRangeError(index, m_size);
      }
      else
        ARCANE_CHECK_AT(index, m_size);
#endif
      return m_data[index];
    }
    constexpr void operator=(ConstArrayView<Int32> v)
    {
      m_data = v.data();
      m_size = v.size();
    }
    // data[NULL_ITEM_LOCAL_ID] must be valid.
    // Therefore, (data-1) must point to a valid address
    void setNull(const Int32* data)
    {
      m_data = data;
      m_size = 0;
    }
    operator ConstArrayView<Int32>() const
    {
      return ConstArrayView<Int32>(m_size, m_data);
    }
    operator SmallSpan<const Int32>() const
    {
      return SmallSpan<const Int32>(m_data, m_size);
    }

   private:

    Int32 m_size;
    const Int32* m_data;
  };

 public:

  enum
  {
    NODE_IDX = 0,
    EDGE_IDX = 1,
    FACE_IDX = 2,
    CELL_IDX = 3,
    HPARENT_IDX = 4,
    HCHILD_IDX = 5,
    MAX_ITEM_KIND = 6
  };

 public:

  static ItemInternalConnectivityList nullInstance;

 public:

  ItemInternalConnectivityList()
  : m_items(nullptr)
  {
    for (Integer i = 0; i < MAX_ITEM_KIND; ++i) {
      m_kind_info[i].m_nb_item_null_data[0] = 0;
      m_kind_info[i].m_nb_item_null_data[1] = 0;
      m_kind_info[i].m_max_nb_item = 0;
    }

    for (Integer i = 0; i < MAX_ITEM_KIND; ++i) {
      m_container[i].m_nb_item.setNull(&m_kind_info[i].m_nb_item_null_data[1]);
      m_container[i].m_offset = ConstArrayView<Int32>{};
    }
  }

 public:

  void updateMeshItemInternalList()
  {
#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
    m_shared_infos.m_node = m_items->m_node_shared_info;
    m_shared_infos.m_edge = m_items->m_edge_shared_info;
    m_shared_infos.m_face = m_items->m_face_shared_info;
    m_shared_infos.m_cell = m_items->m_cell_shared_info;
#endif
  }

 private:

  /*!
   * \brief localId() of the \a index-th entity of type \a item_kind
   * connected to the entity with localid() \a lid.
   */
  constexpr Int32 itemLocalId(Int32 item_kind, Int32 lid, Integer index) const
  {
    return m_container[item_kind].itemLocalId(lid, index);
  }
  //! Number of calls to itemLocalId()
  Int64 nbAccess() const { return 0; }
  //! Number of calls to itemLocalIds()
  Int64 nbAccessAll() const { return 0; }

 private:

  //! Positions the connectivity index array
  void _setConnectivityIndex(Int32 item_kind, ConstArrayView<Int32> v)
  {
    m_container[item_kind].m_indexes = v;
  }
  //! Positions the array containing the connectivity list
  void _setConnectivityList(Int32 item_kind, ArrayView<Int32> v)
  {
    m_container[item_kind].m_list = v;
    m_container[item_kind].m_offset = ConstArrayView<Int32>{};
  }
  //! Positions the array containing the number of connected entities.
  void _setConnectivityNbItem(Int32 item_kind, ConstArrayView<Int32> v)
  {
    m_container[item_kind].m_nb_item = v;
  }
  //! Positions the maximum number of connected entities.
  void _setMaxNbConnectedItem(Int32 item_kind, Int32 v)
  {
    m_kind_info[item_kind].m_max_nb_item = v;
  }

 public:

  //! Connectivity index array for entities of kind \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityIndex(Int32 item_kind) const
  {
    return m_container[item_kind].m_indexes;
  }
  //! Array containing the connectivity list for entities of kind \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityList(Int32 item_kind) const
  {
    return m_container[item_kind].m_list;
  }
  //! Array containing the number of connected entities for entities of kind \a item_kind
  ARCANE_DEPRECATED_REASON("Y2022: Use containerView() instead")
  Int32ConstArrayView connectivityNbItem(Int32 item_kind) const
  {
    return m_container[item_kind].m_nb_item;
  }

 public:

  //! Maximum number of connected entities.
  Int32 maxNbConnectedItem(Int32 item_kind) const
  {
    return m_kind_info[item_kind].m_max_nb_item;
  }

  ItemConnectivityContainerView containerView(Int32 item_kind) const
  {
    return m_container[item_kind].containerView();
  }

 public:

  constexpr ItemBaseBuildInfo nodeBase(Int32 lid, Int32 aindex) const
  {
    return ItemBaseBuildInfo(_nodeLocalIdV2(lid, aindex), A_INTERNAL_SI(node));
  }
  constexpr ItemBaseBuildInfo edgeBase(Int32 lid, Int32 aindex) const
  {
    return ItemBaseBuildInfo(_edgeLocalIdV2(lid, aindex), A_INTERNAL_SI(edge));
  }
  constexpr ItemBaseBuildInfo faceBase(Int32 lid, Int32 aindex) const
  {
    return ItemBaseBuildInfo(_faceLocalIdV2(lid, aindex), A_INTERNAL_SI(face));
  }
  constexpr ItemBaseBuildInfo cellBase(Int32 lid, Int32 aindex) const
  {
    return ItemBaseBuildInfo(_cellLocalIdV2(lid, aindex), A_INTERNAL_SI(cell));
  }
  ItemBaseBuildInfo hParentBase(Int32 lid, Int32 aindex, ItemSharedInfo* isf) const
  {
    return ItemBaseBuildInfo(_hParentLocalIdV2(lid, aindex), isf);
  }
  ItemBaseBuildInfo hChildBase(Int32 lid, Int32 aindex, ItemSharedInfo* isf) const
  {
    return ItemBaseBuildInfo(_hChildLocalIdV2(lid, aindex), isf);
  }

  auto nodeList(Int32 lid) const { return impl::ItemIndexedListView{ A_INTERNAL_SI(node), _itemLocalIdListView(NODE_IDX, lid) }; }
  auto edgeList(Int32 lid) const { return impl::ItemIndexedListView{ A_INTERNAL_SI(edge), _itemLocalIdListView(EDGE_IDX, lid) }; }
  auto faceList(Int32 lid) const { return impl::ItemIndexedListView{ A_INTERNAL_SI(face), _itemLocalIdListView(FACE_IDX, lid) }; }
  auto cellList(Int32 lid) const { return impl::ItemIndexedListView{ A_INTERNAL_SI(cell), _itemLocalIdListView(CELL_IDX, lid) }; }

 private:

  // These 4 methods are still used by ItemBase via internalNodes(), internalEdges(), ...
  // They can be removed when these obsolete methods are removed
  ItemInternalVectorView nodesV2(Int32 lid) const { return { A_INTERNAL_SI(node), _itemLocalIdListView(NODE_IDX, lid) }; }
  ItemInternalVectorView edgesV2(Int32 lid) const { return { A_INTERNAL_SI(edge), _itemLocalIdListView(EDGE_IDX, lid) }; }
  ItemInternalVectorView facesV2(Int32 lid) const { return { A_INTERNAL_SI(face), _itemLocalIdListView(FACE_IDX, lid) }; }
  ItemInternalVectorView cellsV2(Int32 lid) const { return { A_INTERNAL_SI(cell), _itemLocalIdListView(CELL_IDX, lid) }; }

  NodeLocalIdView nodeLocalIdsView(Int32 lid) const { return NodeLocalIdView(_itemLocalIdListView(NODE_IDX, lid)); }
  EdgeLocalIdView edgeLocalIdsView(Int32 lid) const { return EdgeLocalIdView(_itemLocalIdListView(EDGE_IDX, lid)); }
  FaceLocalIdView faceLocalIdsView(Int32 lid) const { return FaceLocalIdView(_itemLocalIdListView(FACE_IDX, lid)); }
  CellLocalIdView cellLocalIdsView(Int32 lid) const { return CellLocalIdView(_itemLocalIdListView(CELL_IDX, lid)); }

 private:

  constexpr Int32 _nodeLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(NODE_IDX, lid, index); }
  constexpr Int32 _edgeLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(EDGE_IDX, lid, index); }
  constexpr Int32 _faceLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(FACE_IDX, lid, index); }
  constexpr Int32 _cellLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(CELL_IDX, lid, index); }
  constexpr Int32 _hParentLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(HPARENT_IDX, lid, index); }
  constexpr Int32 _hChildLocalIdV2(Int32 lid, Int32 index) const { return itemLocalId(HCHILD_IDX, lid, index); }

 private:

  ItemInternal* _nodeV2(Int32 lid, Int32 aindex) const { return m_items->nodes[_nodeLocalIdV2(lid, aindex)]; }
  ItemInternal* _edgeV2(Int32 lid, Int32 aindex) const { return m_items->edges[_edgeLocalIdV2(lid, aindex)]; }
  ItemInternal* _faceV2(Int32 lid, Int32 aindex) const { return m_items->faces[_faceLocalIdV2(lid, aindex)]; }
  ItemInternal* _cellV2(Int32 lid, Int32 aindex) const { return m_items->cells[_cellLocalIdV2(lid, aindex)]; }
  ItemInternal* _hParentV2(Int32 lid, Int32 aindex) const { return m_items->cells[_hParentLocalIdV2(lid, aindex)]; }
  ItemInternal* _hChildV2(Int32 lid, Int32 aindex) const { return m_items->cells[_hChildLocalIdV2(lid, aindex)]; }

 private:

  constexpr Int32 _nbNodeV2(Int32 lid) const { return m_container[NODE_IDX].m_nb_item[lid]; }
  constexpr Int32 _nbEdgeV2(Int32 lid) const { return m_container[EDGE_IDX].m_nb_item[lid]; }
  constexpr Int32 _nbFaceV2(Int32 lid) const { return m_container[FACE_IDX].m_nb_item[lid]; }
  constexpr Int32 _nbCellV2(Int32 lid) const { return m_container[CELL_IDX].m_nb_item[lid]; }
  Int32 _nbHParentV2(Int32 lid) const { return m_container[HPARENT_IDX].m_nb_item[lid]; }
  Int32 _nbHChildrenV2(Int32 lid) const { return m_container[HCHILD_IDX].m_nb_item[lid]; }

 private:

  Int32 _nodeOffset(Int32 lid) const { return m_container[NODE_IDX].itemOffset(lid); }
  Int32 _edgeOffset(Int32 lid) const { return m_container[EDGE_IDX].itemOffset(lid); }
  Int32 _faceOffset(Int32 lid) const { return m_container[FACE_IDX].itemOffset(lid); }
  Int32 _cellOffset(Int32 lid) const { return m_container[CELL_IDX].itemOffset(lid); }
  Int32 _itemOffset(Int32 item_kind, Int32 lid) const { return m_container[item_kind].itemOffset(lid); }

 private:

  impl::ItemLocalIdListContainerView _itemLocalIdListView(Int32 item_kind, Int32 lid) const
  {
    return m_container[item_kind].itemLocalIdListView(lid);
  }

 private:

  // NOTE : eventually, this class will be merged with ItemConnectivityContainerView
  //! Container of views for the connectivity information of a family
  struct Container
  {
    impl::ItemLocalIdListContainerView itemLocalIdListView(Int32 lid) const
    {
      return impl::ItemLocalIdListContainerView(itemLocalIdsData(lid), m_nb_item[lid], itemOffset(lid));
    }
    const Int32* itemLocalIdsData(Int32 lid) const
    {
      return &(m_list[m_indexes[lid]]);
    }
    constexpr Int32 itemLocalId(Int32 lid, Integer index) const
    {
      return m_list[m_indexes[lid] + index] + itemOffset(lid);
    }
    ItemConnectivityContainerView containerView() const
    {
      return ItemConnectivityContainerView(m_list, m_indexes, m_nb_item);
    }
    constexpr Int32 itemOffset([[maybe_unused]] Int32 lid) const
    {
#ifdef ARCANE_USE_OFFSET_FOR_CONNECTIVITY
      return m_offset[lid];
#else
      return 0;
#endif
    }

   public:

    ConstArrayView<Int32> m_indexes;
    Int32View m_nb_item;
    ArrayView<Int32> m_list;
    ConstArrayView<Int32> m_offset;
  };

  struct KindInfo
  {
    Int32 m_max_nb_item;
    Int32 m_nb_item_null_data[2];
  };

 private:

  Container m_container[MAX_ITEM_KIND];
  KindInfo m_kind_info[MAX_ITEM_KIND];

  MeshItemInternalList* m_items;

 private:

#ifdef ARCANE_CONNECTIVITYLIST_USE_OWN_SHAREDINFO
  impl::MeshItemSharedInfoList m_shared_infos;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for mesh entities.
 *
 * This class is internal to %Arcane.
 *
 * This class is normally internal to Arcane, and it is preferable to use
 * specialized versions such as Item, Node, Face, Edge, Cell, Particle,
 * or DoF.
 *
 * Instances of this class are temporary objects that should not be retained
 * between two topological modifications of the mesh if compressions
 * (IItemFamily::compactItems()) occur during these modifications.
 *
 * All methods of this class are read-only and do not allow modification of
 * an entity.
 */
class ARCANE_CORE_EXPORT ItemBase
: public ItemFlags
{
  friend class ::Arcane::ItemInternal;
  friend class ::Arcane::Item;
  friend class ::Arcane::ItemInternalCompatibility;
  friend Arcane::Materials::ConstituentItemSharedInfo;
  friend class ::Arcane::ItemEnumerator;
  friend MutableItemBase;
  // For _internalActiveCells2().
  friend class ::Arcane::Node;
  // For _itemInternal()
  friend class ::Arcane::mesh::ItemFamily;
  friend class ::Arcane::mesh::MeshRefinement;

 private:

  constexpr ItemBase(Int32 local_id, ItemSharedInfo* shared_info)
  : m_local_id(local_id)
  , m_shared_info(shared_info)
  {}

 public:

  ItemBase()
  : m_shared_info(ItemSharedInfo::nullItemSharedInfoPointer)
  {}
  constexpr ItemBase(ItemBaseBuildInfo x)
  : m_local_id(x.m_local_id)
  , m_shared_info(x.m_shared_info)
  {}

 public:

  // TODO: To be removed eventually
  inline ItemBase(ItemInternal* x);

 public:

  //! Local number (in the subdomain) of the entity
  Int32 localId() const { return m_local_id; }
  //! Local number (in the subdomain) of the entity
  inline ItemLocalId itemLocalId() const;
  //! Unique number of the entity
  ItemUniqueId uniqueId() const
  {
#ifdef ARCANE_CHECK
    if (m_local_id != NULL_ITEM_LOCAL_ID)
      arcaneCheckAt((Integer)m_local_id, m_shared_info->m_unique_ids.size());
#endif
    // Do not use the normal accessor because this array can be used for the
    // null cell and in this case m_local_id equals NULL_ITEM_LOCAL_ID (which is negative)
    // which causes an array overflow exception.
    return ItemUniqueId(m_shared_info->m_unique_ids.data()[m_local_id]);
  }

  //! Number of the owning subdomain of the entity
  Int32 owner() const { return m_shared_info->_ownerV2(m_local_id); }

  //! Flags of the entity
  Int32 flags() const { return m_shared_info->_flagsV2(m_local_id); }

  //! Number of nodes of the entity
  Integer nbNode() const { return _connectivity()->_nbNodeV2(m_local_id); }
  //! Number of edges of the entity or number of edges connected to the entity (for nodes)
  Integer nbEdge() const { return _connectivity()->_nbEdgeV2(m_local_id); }
  //! Number of faces of the entity or number of faces connected to the entity (for nodes and edges)
  Integer nbFace() const { return _connectivity()->_nbFaceV2(m_local_id); }
  //! Number of cells connected to the entity (for nodes, edges, and faces)
  Integer nbCell() const { return _connectivity()->_nbCellV2(m_local_id); }
  //! Number of parents for AMR
  Int32 nbHParent() const { return _connectivity()->_nbHParentV2(m_local_id); }
  //! Number of children for AMR
  Int32 nbHChildren() const { return _connectivity()->_nbHChildrenV2(m_local_id); }
  //! Number of parent for sub-meshes
  Integer nbParent() const { return m_shared_info->nbParent(); }

 public:

  //! Type of the entity
  Int16 typeId() const { return m_shared_info->_typeId(m_local_id); }
  //! Type of the entity
  ItemTypeId itemTypeId() const { return ItemTypeId(typeId()); }
  //! Type of the entity.
  ItemTypeInfo* typeInfo() const { return m_shared_info->typeInfoFromId(typeId()); }

  //! @returns the refinement level of the current item. If the parent item is
  //\p NULL, it is conventionally at level 0; otherwise, it is simply at the
  //level of its parent.
  inline Int32 level() const
  {
    //! if I do not have a parent, I was created directly from a file
    //! or by the user, so I am a level 0 item
    if (this->nbHParent() == 0)
      return 0;
    //! otherwise, I am at a higher level than my parent
    return (this->hParentBase(0).level() + 1);
  }

  //! @returns \p true if the item is an ancestor (i.e., has an active
  //! child or an ancestor child), \p false otherwise. Always returns \p false
  //! if AMR is disabled.
  inline bool isAncestor() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return false;
    if (this->hChildBase(0).isActive())
      return true;
    return this->hChildBase(0).isAncestor();
  }
  //! @returns \p true if the item has children (active or not), \p false
  //! otherwise. Always returns \p false if AMR is disabled.
  inline bool hasHChildren() const
  {
    if (this->nbHChildren() == 0) // TODO ? to check!
      return false;
    else
      return true;
  }

  //! @returns \p true if the item is active (i.e., has no active descendants),
  //! \p false otherwise. Note that it is sufficient to check only the first
  //! child. Always returns \p true if AMR is disabled.
  inline bool isActive() const
  {
    if ((flags() & II_Inactive) | (flags() & II_CoarsenInactive))
      return false;
    else
      return true;
  }

  //! @returns \p true if the item is subactive (i.e., not active and has
  //! no descendants), \p false otherwise. Always returns \p false if AMR
  //! is disabled.
  inline bool isSubactive() const
  {
    if (this->isActive())
      return false;
    if (!this->hasHChildren())
      return true;
    return this->hChildBase(0).isSubactive();
  }

  //! Family the entity belongs to
  IItemFamily* family() const { return m_shared_info->m_item_family; }
  //! Kind of the entity
  eItemKind kind() const { return m_shared_info->m_item_kind; }
  //! True if the entity is the null entity
  bool null() const { return m_local_id == NULL_ITEM_LOCAL_ID; }
  //! True if the entity is the null entity
  bool isNull() const { return m_local_id == NULL_ITEM_LOCAL_ID; }
  //! True if the entity belongs to the subdomain
  bool isOwn() const { return ItemFlags::isOwn(flags()); }
  /*!
   * \brief True if the entity is shared by other subdomains.
   *
   * This method is only relevant if the connectivity information has
   * been calculated.
   */
  bool isShared() const { return ItemFlags::isShared(flags()); }

  //! True if the entity is suppressed
  bool isSuppressed() const { return (flags() & II_Suppressed) != 0; }
  //! True if the entity is detached
  bool isDetached() const { return (flags() & II_Detached) != 0; }

  //! \a true if the entity is on the boundary
  bool isBoundary() const { return ItemFlags::isBoundary(flags()); }
  //! Cell connected to the entity if the entity is a boundary entity (0 if none)
  ItemBase boundaryCell() const { return (flags() & II_Boundary) ? cellBase(0) : ItemBase(); }
  //! Cell behind the entity (nullItem() if none)
  ItemBase backCell() const
  {
    if (flags() & II_HasBackCell)
      return cellBase((flags() & II_BackCellIsFirst) ? 0 : 1);
    return {};
  }
  //! Cell behind the entity (NULL_ITEM_LOCAL_ID if none)
  Int32 backCellId() const
  {
    if (flags() & II_HasBackCell)
      return cellId((flags() & II_BackCellIsFirst) ? 0 : 1);
    return NULL_ITEM_LOCAL_ID;
  }
  //! Cell in front of the entity (nullItem() if none)
  ItemBase frontCell() const
  {
    if (flags() & II_HasFrontCell)
      return cellBase((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return {};
  }
  //! Cell in front of the entity (NULL_ITEM_LOCAL_ID if none)
  Int32 frontCellId() const
  {
    if (flags() & II_HasFrontCell)
      return cellId((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return NULL_ITEM_LOCAL_ID;
  }
  ItemBase masterFace() const
  {
    if (flags() & II_SlaveFace)
      return faceBase(0);
    return {};
  }
  //! \a true if it is the master face of an interface
  inline bool isMasterFace() const { return flags() & II_MasterFace; }

  //! \a true if it is a slave face of an interface
  inline bool isSlaveFace() const { return flags() & II_SlaveFace; }

  Int32 parentId(Integer index) const { return m_shared_info->_parentLocalIdV2(m_local_id, index); }

  //@{
  Int32 nodeId(Integer index) const { return _connectivity()->_nodeLocalIdV2(m_local_id, index); }
  Int32 edgeId(Integer index) const { return _connectivity()->_edgeLocalIdV2(m_local_id, index); }
  Int32 faceId(Integer index) const { return _connectivity()->_faceLocalIdV2(m_local_id, index); }
  Int32 cellId(Integer index) const { return _connectivity()->_cellLocalIdV2(m_local_id, index); }
  Int32 hParentId(Int32 index) const { return _connectivity()->_hParentLocalIdV2(m_local_id, index); }
  Int32 hChildId(Int32 index) const { return _connectivity()->_hChildLocalIdV2(m_local_id, index); }
  //@}

  /*!
   * \brief Methods using the new connectivities to access connectivity
   * information. Should not be used outside of Arcane.
   *
   * \warning These methods must only be called on entities that possess
   * the associated connectivity AND are in the new format. For example,
   * this does not work on Cell->Cell because there is no cell/cell
   * connectivity. Misuse results in an array overflow.
   */
  //@{
  ARCANE_DEPRECATED_REASON("Y2023: Use nodeList() instead.")
  ItemInternalVectorView internalNodes() const { return _connectivity()->nodesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeList() instead.")
  ItemInternalVectorView internalEdges() const { return _connectivity()->edgesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceList() instead.")
  ItemInternalVectorView internalFaces() const { return _connectivity()->facesV2(m_local_id); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellList() instead.")
  ItemInternalVectorView internalCells() const { return _connectivity()->cellsV2(m_local_id); }
  //@}

  /*!
   * \brief Methods using the new connectivities to access connectivity
   * information. Should not be used outside of Arcane.
   *
   * \warning These methods must only be called on entities that possess
   * the associated connectivity. For example, this does not work on
   * Cell->Cell because there is no cell/cell connectivity. Misuse results
   * in an array overflow.
   */
  //@{
  impl::ItemIndexedListView<DynExtent> nodeList() const { return _connectivity()->nodeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> edgeList() const { return _connectivity()->edgeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> faceList() const { return _connectivity()->faceList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> cellList() const { return _connectivity()->cellList(m_local_id); }

  impl::ItemIndexedListView<DynExtent> itemList(Node*) const { return nodeList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Edge*) const { return edgeList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Face*) const { return faceList(); }
  impl::ItemIndexedListView<DynExtent> itemList(Cell*) const { return cellList(); }
  //@}

  ItemBase nodeBase(Int32 index) const { return _connectivity()->nodeBase(m_local_id, index); }
  ItemBase edgeBase(Int32 index) const { return _connectivity()->edgeBase(m_local_id, index); }
  ItemBase faceBase(Int32 index) const { return _connectivity()->faceBase(m_local_id, index); }
  ItemBase cellBase(Int32 index) const { return _connectivity()->cellBase(m_local_id, index); }
  ItemBase hParentBase(Int32 index) const { return _connectivity()->hParentBase(m_local_id, index, m_shared_info); }
  ItemBase hChildBase(Int32 index) const { return _connectivity()->hChildBase(m_local_id, index, m_shared_info); }
  inline ItemBase parentBase(Int32 index) const;

  //! Returns whether the flags \a flags are set for the entity
  bool hasFlags(Int32 flags) const { return (this->flags() & flags); }

 public:

  /*!
   * @returns the rank of the child \p (iitem).
   * example: if rank = m_internal->whichChildAmI(iitem); then
   * m_internal->hChild(rank) would be iitem;
   */
  Int32 whichChildAmI(Int32 local_id) const;

 public:

  ItemBase topHParentBase() const;

 public:

  //! Mutable interface of this entity
  inline MutableItemBase toMutable();

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane.")
  inline ItemInternal* itemInternal() const;

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane.")
  ItemInternalVectorView _internalActiveCells(Int32Array& local_ids) const
  {
    return _internalActiveCells2(local_ids);
  }

 private:

  Int32 _nbLinearNode() const;

 private:

  /*!
   * \brief Local number (in the subdomain) of the entity.
   *
   * For performance reasons, the local number must be
   * the first field of the class.
   */
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;

  //! Field used only to explicitly manage alignment
  Int32 m_padding = 0;

  //! Shared info between all entities with the same characteristics
  ItemSharedInfo* m_shared_info = nullptr;

 private:

  ItemInternalConnectivityList* _connectivity() const
  {
    return m_shared_info->m_connectivity;
  }
  void _setFromInternal(ItemBase* rhs)
  {
    m_local_id = rhs->m_local_id;
    m_shared_info = rhs->m_shared_info;
  }
  void _setFromInternal(const ItemBase& rhs)
  {
    m_local_id = rhs.m_local_id;
    m_shared_info = rhs.m_shared_info;
  }
  ItemInternalVectorView _internalActiveCells2(Int32Array& local_ids) const;
  inline ItemInternal* _itemInternal() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Methods allowing modification of ItemBase.
 *
 * These methods are internal to Arcane.
 */
class ARCANE_CORE_EXPORT MutableItemBase
: public ItemBase
{
  friend class ::Arcane::Item;
  friend ItemBase;
  // For _setFaceBackAndFrontCell()
  friend Arcane::mesh::FaceFamily;

 private:

  MutableItemBase(Int32 local_id, ItemSharedInfo* shared_info)
  : ItemBase(local_id, shared_info)
  {}

 public:

  MutableItemBase() = default;
  MutableItemBase(ItemBaseBuildInfo x)
  : ItemBase(x)
  {}
  explicit MutableItemBase(const ItemBase& x)
  : ItemBase(x)
  {}

 public:

  // TODO: To be removed eventually
  inline MutableItemBase(ItemInternal* x);

 public:

  void setUniqueId(Int64 uid)
  {
    _checkUniqueId(uid);
    m_shared_info->m_unique_ids[m_local_id] = uid;
  }

  //! Nullifies the uniqueId to the value NULL_ITEM_UNIQUE_ID
  /*! Checks that the value to be canceled is valid in ARCANE_CHECK mode */
  void unsetUniqueId();

  /*!
   * \brief Sets the sub-domain number of the entity owner.

    \a current_sub_domain is the sub-domain number calling this operation.

    After calling this function, you must update the mesh to which this entity
    belongs by calling the IMesh::notifyOwnItemsChanged() method. It is not
    necessary to call this method for every call to setOwn. Only one
    call after all modifications is necessary.
  */
  void setOwner(Integer suid, Int32 current_sub_domain)
  {
    m_shared_info->_setOwnerV2(m_local_id, suid);
    int f = flags();
    if (suid == current_sub_domain)
      f |= II_Own;
    else
      f &= ~II_Own;
    setFlags(f);
  }

  //! Sets the entity flags
  void setFlags(Int32 f) { m_shared_info->_setFlagsV2(m_local_id, f); }

  //! Adds the flags \a added_flags to those of the entity
  void addFlags(Int32 added_flags)
  {
    Int32 f = this->flags();
    f |= added_flags;
    this->setFlags(f);
  }

  //! Removes the flags \a removed_flags from those of the entity
  void removeFlags(Int32 removed_flags)
  {
    Int32 f = this->flags();
    f &= ~removed_flags;
    this->setFlags(f);
  }

  //! Sets the detached state of the entity
  void setDetached(bool v)
  {
    int f = flags();
    if (v)
      f |= II_Detached;
    else
      f &= ~II_Detached;
    setFlags(f);
  }

  void reinitialize(Int64 uid, Int32 aowner, Int32 owner_rank)
  {
    setUniqueId(uid);
    setFlags(0);
    setOwner(aowner, owner_rank);
  }

  void setLocalId(Int32 local_id)
  {
    m_local_id = local_id;
  }

  //! Sets the \a i-th parent (currently aindex must be 0)
  void setParent(Int32 aindex, Int32 parent_local_id)
  {
    m_shared_info->_setParentV2(m_local_id, aindex, parent_local_id);
  }

 private:

  void _setFaceBackAndFrontCells(Int32 back_cell_lid, Int32 front_cell_lid);

  void _checkUniqueId(Int64 new_uid) const;

  inline void _setFaceInfos(Int32 mod_flags);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal structure of a mesh entity.

 This instance contains the internal structure of a mesh entity.
 It should only be manipulated by those who know what they are doing...

 To use an entity, you must use the Item class or one of its derived classes.

 In general, the mesh (IMesh) to which the entity belongs maintains
 different structures allowing the mesh to be manipulated. These structures
 are often recalculated dynamically when necessary (lazy evaluation). This is
 the case, for example, with sub-domain specific entity groups or the table
 for converting global numbers to local numbers. This is why it is essential
 when performing a series of modifications to instances of this class to
 notify the mesh of the changes made.
 */
class ARCANE_CORE_EXPORT ItemInternal
: public impl::MutableItemBase
{
  // For access to _setSharedInfo()
  friend class mesh::DynamicMeshKindInfos;
  friend class mesh::ItemFamily;

 public:

  //! Null entity
  static ItemInternal nullItemInternal;
  static ItemInternal* nullItem() { return &nullItemInternal; }

 public:

  // You must use the corresponding method from ItemBase

  //! Connected cell to the entity if the entity is a boundary entity (0 if none)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::boundaryCell() instead.")
  ItemInternal* boundaryCell() const { return (flags() & II_Boundary) ? _internalCell(0) : nullItem(); }
  //! Cell behind the entity (nullItem() if none)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::backCell() instead.")
  ItemInternal* backCell() const
  {
    if (flags() & II_HasBackCell)
      return _internalCell((flags() & II_BackCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  //! Cell in front of the entity (nullItem() if none)
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::frontCell() instead.")
  ItemInternal* frontCell() const
  {
    if (flags() & II_HasFrontCell)
      return _internalCell((flags() & II_FrontCellIsFirst) ? 0 : 1);
    return nullItem();
  }
  ARCANE_DEPRECATED_REASON("Y2023: use ItemBase::masterFace() instead.")
  ItemInternal* masterFace() const
  {
    if (flags() & II_SlaveFace)
      return _internalFace(0);
    return nullItem();
  }

 public:

  //! Shared information of the entity.
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  ItemSharedInfo* sharedInfo() const { return m_shared_info; }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Node*) const { return nodeList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Edge*) const { return edgeList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Face*) const { return faceList(); }
  ARCANE_DEPRECATED_REASON("Y2023: Use itemList() instead.")
  ItemInternalVectorView internalItems(Cell*) const { return cellList(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use nodeBase() instead.")
  ItemInternal* internalNode(Int32 index) const { return _connectivity()->_nodeV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeBase() instead.")
  ItemInternal* internalEdge(Int32 index) const { return _connectivity()->_edgeV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceBase() instead.")
  ItemInternal* internalFace(Int32 index) const { return _connectivity()->_faceV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellBase() instead.")
  ItemInternal* internalCell(Int32 index) const { return _connectivity()->_cellV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use hParentBase() instead.")
  ItemInternal* internalHParent(Int32 index) const { return _connectivity()->_hParentV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use hChildBase() instead.")
  ItemInternal* internalHChild(Int32 index) const { return _connectivity()->_hChildV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use parentBase() instead.")
  ItemInternal* parent(Integer index) const { return m_shared_info->_parentV2(m_local_id, index); }

 public:

  const ItemInternal* topHParent() const;
  ItemInternal* topHParent();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always returns 0")
  Int32 dataIndex() { return 0; }

 public:

  /*!
   * \brief Pointer to the list of parents.
   *
   * Since currently only one level is supported, it is only allowed
   * to call parentPtr()[0]. This does not allow any verification,
   * so it is preferable to use parentId() or setParent() instead.
   *
   * As of July 2022, this method is no longer used in Arcane, so if
   * no code uses it (which should be the case since it is an internal
   * method), we can quickly remove it.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use parentId() or setParent() instead")
  Int32* parentPtr() { return m_shared_info->_parentPtr(m_local_id); }

  /*!
   * @returns the rank of the child \p (iitem).
   * Example: if rank = m_internal->whichChildAmI(iitem); then
   * m_internal->hChild(rank) would be iitem;
   */
  Int32 whichChildAmI(const ItemInternal* iitem) const;

 public:

  //! Memory required to store the entity information
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Integer neededMemory() const { return 0; }

  //! Minimum memory required to store the entity information (without buffer)
  ARCANE_DEPRECATED_REASON("Y2022: This method always return 0")
  constexpr Integer minimumNeededMemory() const { return 0; }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use nodeId() instead")
  Int32 nodeLocalId(Integer index) { return _connectivity()->_nodeLocalIdV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use edgeId() instead")
  Int32 edgeLocalId(Integer index) { return _connectivity()->_edgeLocalIdV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use faceId() instead")
  Int32 faceLocalId(Integer index) { return _connectivity()->_faceLocalIdV2(m_local_id, index); }
  ARCANE_DEPRECATED_REASON("Y2023: Use cellId() instead")
  Int32 cellLocalId(Integer index) { return _connectivity()->_cellLocalIdV2(m_local_id, index); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: This method always throws an exception.")
  void setDataIndex(Integer);

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  void setSharedInfo(ItemSharedInfo* shared_infos, ItemTypeId type_id)
  {
    _setSharedInfo(shared_infos, type_id);
  }

 public:

  //! \internal
  typedef ItemInternal* ItemInternalPtr;

  //! \internal
  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane and should not be used.")
  static ItemSharedInfo* _getSharedInfo(const ItemInternalPtr* items)
  {
    return ((items) ? items[0]->m_shared_info : ItemSharedInfo::nullInstance());
  }

 private:

  void _setSharedInfo(ItemSharedInfo* shared_infos, ItemTypeId type_id)
  {
    m_shared_info = shared_infos;
    shared_infos->_setTypeId(m_local_id, type_id.typeId());
  }

  ItemInternal* _internalFace(Int32 index) const { return _connectivity()->_faceV2(m_local_id, index); }
  ItemInternal* _internalCell(Int32 index) const { return _connectivity()->_cellV2(m_local_id, index); }
  ItemInternal* _internalHParent(Int32 index) const { return _connectivity()->_hParentV2(m_local_id, index); }
  ItemInternal* _internalHChild(Int32 index) const { return _connectivity()->_hChildV2(m_local_id, index); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemBase::
ItemBase(ItemInternal* x)
: m_local_id(x->m_local_id)
, m_shared_info(x->m_shared_info)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableItemBase::
MutableItemBase(ItemInternal* x)
: ItemBase(x)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(ItemInternal* item)
: m_local_id(item->localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: add type check
template <typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemInternal* item)
: ItemLocalId(item->localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* impl::ItemBase::
itemInternal() const
{
  if (m_local_id != NULL_ITEM_LOCAL_ID)
    return m_shared_info->m_items_internal[m_local_id];
  return ItemInternal::nullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* impl::ItemBase::
_itemInternal() const
{
  if (m_local_id != NULL_ITEM_LOCAL_ID)
    return m_shared_info->m_items_internal[m_local_id];
  return ItemInternal::nullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline impl::ItemBase impl::ItemBase::
parentBase(Int32 index) const
{
  return ItemBase(m_shared_info->_parentV2(m_local_id, index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline impl::MutableItemBase impl::ItemBase::
toMutable()
{
  return MutableItemBase(m_local_id, m_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId impl::ItemBase::
itemLocalId() const
{
  return ItemLocalId(m_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Methods for conversions between different entity management classes
 *
 * This class is temporary and internal to Arcane. Only 'friend' classes
 * can use it.
 */
class ItemInternalCompatibility
{
  friend class SimdItemBase;
  friend class SimdItemDirectBase;
  friend class SimdItem;
  friend class SimdItemEnumeratorBase;
  friend class ItemVectorView;
  template <typename T> friend class ItemEnumeratorBaseT;
  friend class mesh::DynamicMeshKindInfos;
  friend class TotalviewAdapter;
  template <int Extent> friend class ItemConnectedListView;

 private:

  //! \internal
  typedef ItemInternal* ItemInternalPtr;
  static ItemSharedInfo* _getSharedInfo(const ItemInternal* item)
  {
    return item->m_shared_info;
  }
  static ItemSharedInfo* _getSharedInfo(const ItemInternalPtr* items, Int32 count)
  {
    return ((items && count > 0) ? items[0]->m_shared_info : ItemSharedInfo::nullInstance());
  }
  static const ItemInternalPtr* _getItemInternalPtr(ItemSharedInfo* shared_info)
  {
    ARCANE_CHECK_PTR(shared_info);
    return shared_info->m_items_internal.data();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
