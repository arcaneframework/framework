// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Item.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Information about mesh elements.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEM_H
#define ARCANE_CORE_ITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/ItemLocalId.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro to check in Check mode that conversions between the
// entity kinds are correct.
#ifdef ARCANE_CHECK
#define ARCANE_CHECK_KIND(type) _checkKind(type())
#else
#define ARCANE_CHECK_KIND(type)
#endif

#ifdef ARCANE_CHECK
#define ARCANE_WANT_ITEM_STAT
#endif

#ifdef ARCANE_WANT_ITEM_STAT
#define ARCANE_ITEM_ADD_STAT(var) ++var
#else
#define ARCANE_ITEM_ADD_STAT(var)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for a mesh element.
 *
 * \ingroup Mesh

 Mesh elements are nodes (Node), cells (Cell),
 faces (Face), edges (Edge), particles (Particle) or degrees of freedom (DoF).
 Each of its elements is described
 in the corresponding derived class.

 This class and its derived classes are lightweight objects that are used by
 value rather than by reference and should not be kept between two
 modifications of the family (IItemFamily) they are associated with.

 Regardless of its type, a mesh element has a unique identifier
 (localId()) for its type and local to the managed subdomain and a unique identifier
 (uniqueId()) for its type across the entire domain. The numbering is
 <b>continuous</b> and starts at <b>0</b>. The local identifier is used, for example,
 to access variables or for connectivity.

 For example, if a mesh has 2 hexahedral cells that join
 by a face, there are 12 nodes, 11 faces, and 2 cells. In this case, the first
 node will have identifier 0, the second 1, and so on up to 11. The
 first face will have identifier 0, the second 1, and so on
 up to 10.

 There is an entity corresponding to a null object. It is the only one
 for which null() is true. No operation other than calling null() and comparison
 operations is valid on the null entity.
 */
class ARCANE_CORE_EXPORT Item
{
  // To access private constructors
  friend class ItemEnumeratorBaseT<Item>;
  friend class ItemConnectedEnumeratorBaseT<Item>;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemVectorViewConstIterator;
  friend class ItemConnectedListViewConstIterator;
  friend class SimdItem;
  friend class SimdItemEnumeratorBase;
  friend class ItemInfoListView;
  friend class ItemLocalIdToItemConverter;
  template <typename ItemType> friend class ItemLocalIdToItemConverterT;
  friend class ItemPairEnumerator;
  template <int Extent> friend class ItemConnectedListView;
  template <typename ItemType> friend class ItemEnumeratorBaseT;

  // To access _internal()
  friend class ItemCompatibility;

 public:

  typedef ItemInternal* ItemInternalPtr;

  //! Type of localId()
  typedef ItemLocalId LocalIdType;

  using ItemBase = impl::ItemBase;

 public:

  /*!
   * \brief Index of an Item in a variable.
   * \deprecated
   */
  class Index
  {
    // TODO Deprecate when we remove
    // the obsolete derived classes.
    // We cannot do this before because it generates too many
    // compilation warnings.
   public:

    Index()
    : m_local_id(NULL_ITEM_LOCAL_ID)
    {}
    explicit Index(Int32 id)
    : m_local_id(id)
    {}
    Index(Item item)
    : m_local_id(item.localId())
    {}
    operator ItemLocalId() const { return ItemLocalId{ m_local_id }; }

   public:

    Int32 localId() const { return m_local_id; }

   private:

    Int32 m_local_id;
  };

 public:

  /*!
   * \brief Element types.
   *
   * Type values must range from 0 to #NB_TYPE in steps of 1.
   *
   * \deprecated. Use types defined in ArcaneTypes.h
   */
  enum
  {
    Unknown ARCANE_DEPRECATED_REASON("Use 'IT_NullType' instead") = IT_NullType, //!< Null type element
    Vertex ARCANE_DEPRECATED_REASON("Use 'IT_Vertex' instead") = IT_Vertex, //!< Node type element (1 vertex 1D, 2D and 3D)
    Bar2 ARCANE_DEPRECATED_REASON("Use 'IT_Line2' instead") = IT_Line2, //!< Edge type element (2 vertices, 1D, 2D and 3D)
    Tri3 ARCANE_DEPRECATED_REASON("Use 'IT_Triangle3' instead") = IT_Triangle3, //!< Triangle type element (3 vertices, 2D)
    Quad4 ARCANE_DEPRECATED_REASON("Use 'IT_Quad4' instead") = IT_Quad4, //!< Quad type element (4 vertices, 2D)
    Pentagon5 ARCANE_DEPRECATED_REASON("Use 'IT_Pentagon5' instead") = IT_Pentagon5, //!< Pentagon type element (5 vertices, 2D)
    Hexagon6 ARCANE_DEPRECATED_REASON("Use 'IT_Hexagon6' instead") = IT_Hexagon6, //!< Hexagon type element (6 vertices, 2D)
    Tetra ARCANE_DEPRECATED_REASON("Use 'IT_Tetraedron4' instead") = IT_Tetraedron4, //!< Tetrahedron type element (4 vertices, 3D)
    Pyramid ARCANE_DEPRECATED_REASON("Use 'IT_Pyramid5' instead") = IT_Pyramid5, //!< Pyramid type element (5 vertices, 3D)
    Penta ARCANE_DEPRECATED_REASON("Use 'IT_Pentaedron6' instead") = IT_Pentaedron6, //!< Pentahedron type element (6 vertices, 3D)
    Hexa ARCANE_DEPRECATED_REASON("Use 'IT_Hexaedron8' instead") = IT_Hexaedron8, //!< Hexahedron type element (8 vertices, 3D)
    Wedge7 ARCANE_DEPRECATED_REASON("Use 'IT_Heptaedron10' instead") = IT_Heptaedron10, //!< Prism type element with 7 faces (pentagonal base)
    Wedge8 ARCANE_DEPRECATED_REASON("Use 'IT_Octaedron12' instead") = IT_Octaedron12 //!< Prism type element with 8 faces (hexagonal base)
    // Reduced to minimum for compatibility.
  };

  //! Null element index
  static const Int32 NULL_ELEMENT = NULL_ITEM_ID;

  //! Mesh type name \a cell_type
  ARCCORE_DEPRECATED_2021("Use ItemTypeMng::typeName() instead")
  static String typeName(Int32 type);

 protected:

  //! Constructor reserved for enumerators
  constexpr ARCCORE_HOST_DEVICE Item(Int32 local_id, ItemSharedInfo* shared_info)
  : m_shared_info(shared_info)
  , m_local_id(local_id)
  {}

 public:

  //! Creation of a null mesh entity
  Item() = default;

  //! Constructs a reference to the \a internal entity
  //ARCANE_DEPRECATED_REASON("Remove this overload")
  Item(ItemInternal* ainternal)
  {
    ARCANE_CHECK_PTR(ainternal);
    m_shared_info = ainternal->m_shared_info;
    m_local_id = ainternal->m_local_id;
    ARCANE_ITEM_ADD_STAT(m_nb_created_from_internal);
  }

  // NOTE: For the following constructor; it is essential to use
  // const& to avoid ambiguity with the copy constructor
  //! Constructs a reference to the \a abase entity
  constexpr ARCCORE_HOST_DEVICE Item(const ItemBase& abase)
  : m_shared_info(abase.m_shared_info)
  , m_local_id(abase.m_local_id)
  {
  }

  //! Constructs a reference to the \a internal entity
  Item(const ItemInternalPtr* internals, Int32 local_id)
  : Item(local_id, internals[local_id]->m_shared_info)
  {
    ARCANE_ITEM_ADD_STAT(m_nb_created_from_internalptr);
  }

  //! Copy operator
  Item& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! \a true if the entity is null (i.e. not connected to the mesh)
  constexpr bool null() const { return m_local_id == NULL_ITEM_ID; }

  //! Local identifier of the entity in the processor subdomain
  constexpr Int32 localId() const { return m_local_id; }

  //! Local identifier of the entity in the processor subdomain
  constexpr ItemLocalId itemLocalId() const { return ItemLocalId{ m_local_id }; }

  //! Unique identifier across all domains
  ItemUniqueId uniqueId() const
  {
#ifdef ARCANE_CHECK
    if (m_local_id != NULL_ITEM_LOCAL_ID)
      arcaneCheckAt((Integer)m_local_id, m_shared_info->m_unique_ids.size());
#endif
    // Do not use the normal accessor because this array may be used for the
    // null mesh and in this case m_local_id equals NULL_ITEM_LOCAL_ID (which is negative)
    // which causes an exception for array overflow.
    return ItemUniqueId(m_shared_info->m_unique_ids.data()[m_local_id]);
  }

  //! Owner subdomain number of the entity
  Int32 owner() const { return m_shared_info->_ownerV2(m_local_id); }

  //! Entity type
  Int16 type() const { return m_shared_info->_typeId(m_local_id); }

  //! Entity type
  ItemTypeId itemTypeId() const { return ItemTypeId(type()); }

  //! Family from which the entity originates
  IItemFamily* itemFamily() const { return m_shared_info->m_item_family; }

  //! Entity kind
  constexpr eItemKind kind() const { return m_shared_info->m_item_kind; }

  //! \a true if the entity belongs to the subdomain
  constexpr bool isOwn() const { return (_flags() & ItemFlags::II_Own) != 0; }

  /*!
   * \brief True if the entity is shared by other subdomains.
   *
   * An entity is considered shared if and only if
   * isOwn() is true and it is ghost for one or more
   * other subdomains.
   *
   * This method is only relevant if the connectivity information
   * has been calculated (by calling IItemFamily::computeSynchronizeInfos()).
   */
  bool isShared() const { return (_flags() & ItemFlags::II_Shared) != 0; }

  //! Converts the entity to the \a ItemWithNodes kind.
  inline ItemWithNodes toItemWithNodes() const;
  //! Converts the entity to the \a Node kind.
  inline Node toNode() const;
  //! Converts the entity to the \a Cell kind.
  inline Cell toCell() const;
  //! Converts the entity to the \a Edge kind.
  inline Edge toEdge() const;
  //! Converts the entity to the \a Face kind.
  inline Face toFace() const;
  //! Converts the entity to the \a Particle kind.
  inline Particle toParticle() const;
  //! Converts the entity to the \a DoF kind.
  inline DoF toDoF() const;

  //! Number of parents for submeshes
  Int32 nbParent() const { return _nbParent(); }

  //! i-th parent for submeshes
  Item parent(Int32 i) const { return m_shared_info->_parentV2(m_local_id, i); }

  //! first parent for submeshes
  Item parent() const { return m_shared_info->_parentV2(m_local_id, 0); }

 public:

  //! \a true if the entity is of the \a ItemWithNodes kind.
  constexpr bool isItemWithNodes() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Edge || ik == IK_Face || ik == IK_Cell);
  }

  //! \a true if the entity is of the \a Node kind.
  constexpr bool isNode() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Node);
  }
  //! \a true if the entity is of the \a Cell kind.
  constexpr bool isCell() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Cell);
  }
  //! \a true if the entity is of the \a Edge kind.
  constexpr bool isEdge() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Edge);
  }
  //! \a true if the entity is of the \a Face kind.
  constexpr bool isFace() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Face);
  }
  //! \a true if the entity is of the \a Particle kind.
  constexpr bool isParticle() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_Particle);
  }
  constexpr //! \a true if the entity is of the \a DoF kind
  bool isDoF() const
  {
    eItemKind ik = kind();
    return (ik == IK_Unknown || ik == IK_DoF);
  }

  //! Returns if the \a flags are set for the entity
  constexpr bool hasFlags(Int32 flags) const { return (_flags() & flags); }

  //! Entity flags
  constexpr Int32 flags() const { return m_shared_info->_flagsV2(m_local_id); }

 public:

  /*!
   * \brief Internal part of the entity.
   *
   * \warning The internal part of the entity should only be modified by
   * those who know what they are doing.
   * \deprecated Use itemBase() or mutableItemBase() instead for
   * cases where the returned instance is not kept.
   */
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. use itemBase() or mutableItemBase() instead")
  ItemInternal* internal() const
  {
    if (m_local_id != NULL_ITEM_LOCAL_ID)
      return m_shared_info->m_items_internal[m_local_id];
    return ItemInternal::nullItem();
  }

 public:

  /*!
   * \brief Internal part of the entity.
   *
   * \warning The internal part of the entity should only be modified by
   * those who know what they are doing.
   */
  impl::ItemBase itemBase() const
  {
    return impl::ItemBase(m_local_id, m_shared_info);
  }

  /*!
   * \brief Mutable internal part of the entity.
   *
   * \warning The internal part of the entity should only be modified by
   * those who know what they are doing.
   */
  impl::MutableItemBase mutableItemBase() const
  {
    return impl::MutableItemBase(m_local_id, m_shared_info);
  }

  /*!
   * \brief Information about the entity type.
   *
   * This method allows obtaining information concerning
   * a given entity type, such as the local numbering
   * of its faces or edges.
   */
  const ItemTypeInfo* typeInfo() const { return m_shared_info->typeInfoFromId(type()); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Item* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Item* operator->() const { return this; }

 private:

  //! Shared information among all entities with the same characteristics
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullItemSharedInfoPointer;

 protected:

  /*!
   * \brief Local number (in the subdomain) of the entity.
   *
   * For performance reasons, the local number must be
   * the first field of the class.
   */
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;

 protected:

  constexpr void _checkKind(bool is_valid) const
  {
    if (!is_valid)
      _badConversion();
  }
  [[noreturn]] void _badConversion() const;
  void _set(ItemInternal* ainternal)
  {
    _setFromInternal(ainternal);
  }
  constexpr void _set(const Item& rhs)
  {
    _setFromItem(rhs);
  }

 protected:

  //! Entity flags
  constexpr Int32 _flags() const { return m_shared_info->_flagsV2(m_local_id); }
  //! Number of nodes of the entity
  constexpr Integer _nbNode() const { return _connectivity()->_nbNodeV2(m_local_id); }
  //! Number of edges of the entity or number of edges connected to the entity (for nodes)
  constexpr Integer _nbEdge() const { return _connectivity()->_nbEdgeV2(m_local_id); }
  //! Number of faces of the entity or number of faces connected to the entity (for nodes and edges)
  constexpr Integer _nbFace() const { return _connectivity()->_nbFaceV2(m_local_id); }
  //! Number of cells connected to the entity (for nodes, edges and faces)
  constexpr Integer _nbCell() const { return _connectivity()->_nbCellV2(m_local_id); }
  //! Number of parents for AMR
  Int32 _nbHParent() const { return _connectivity()->_nbHParentV2(m_local_id); }
  //! Number of children for AMR
  Int32 _nbHChildren() const { return _connectivity()->_nbHChildrenV2(m_local_id); }
  //! Number of parents for submeshes
  Integer _nbParent() const { return m_shared_info->nbParent(); }
  constexpr NodeLocalId _nodeId(Int32 index) const { return NodeLocalId(_connectivity()->_nodeLocalIdV2(m_local_id, index)); }
  constexpr EdgeLocalId _edgeId(Int32 index) const { return EdgeLocalId(_connectivity()->_edgeLocalIdV2(m_local_id, index)); }
  constexpr FaceLocalId _faceId(Int32 index) const { return FaceLocalId(_connectivity()->_faceLocalIdV2(m_local_id, index)); }
  constexpr CellLocalId _cellId(Int32 index) const { return CellLocalId(_connectivity()->_cellLocalIdV2(m_local_id, index)); }
  Int32 _hParentId(Int32 index) const { return _connectivity()->_hParentLocalIdV2(m_local_id, index); }
  Int32 _hChildId(Int32 index) const { return _connectivity()->_hChildLocalIdV2(m_local_id, index); }
  impl::ItemIndexedListView<DynExtent> _nodeList() const { return _connectivity()->nodeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _edgeList() const { return _connectivity()->edgeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _faceList() const { return _connectivity()->faceList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _cellList() const { return _connectivity()->cellList(m_local_id); }
  NodeLocalIdView _nodeIds() const { return _connectivity()->nodeLocalIdsView(m_local_id); }
  EdgeLocalIdView _edgeIds() const { return _connectivity()->edgeLocalIdsView(m_local_id); }
  FaceLocalIdView _faceIds() const { return _connectivity()->faceLocalIdsView(m_local_id); }
  CellLocalIdView _cellIds() const { return _connectivity()->cellLocalIdsView(m_local_id); }

  constexpr inline Node _node(Int32 index) const;
  constexpr inline Edge _edge(Int32 index) const;
  constexpr inline Face _face(Int32 index) const;
  constexpr inline Cell _cell(Int32 index) const;

  ItemBase _hParentBase(Int32 index) const { return _connectivity()->hParentBase(m_local_id, index, m_shared_info); }
  ItemBase _hChildBase(Int32 index) const { return _connectivity()->hChildBase(m_local_id, index, m_shared_info); }
  ItemBase _toItemBase() const { return ItemBase(m_local_id, m_shared_info); }

  //! Number of nodes of the entity
  Int32 _nbLinearNode() const { return itemBase()._nbLinearNode(); }

 private:

  constexpr ItemInternalConnectivityList* _connectivity() const
  {
    return m_shared_info->m_connectivity;
  }
  void _setFromInternal(ItemBase* rhs)
  {
    ARCANE_ITEM_ADD_STAT(m_nb_set_from_internal);
    m_local_id = rhs->m_local_id;
    m_shared_info = rhs->m_shared_info;
  }
  constexpr void _setFromItem(const Item& rhs)
  {
    m_local_id = rhs.m_local_id;
    m_shared_info = rhs.m_shared_info;
  }

 public:

  static void dumpStats(ITraceMng* tm);
  static void resetStats();

 private:

  static std::atomic<int> m_nb_created_from_internal;
  static std::atomic<int> m_nb_created_from_internalptr;
  static std::atomic<int> m_nb_set_from_internal;

 private:

  ItemInternal* _internal() const
  {
    if (m_local_id != NULL_ITEM_LOCAL_ID)
      return m_shared_info->m_items_internal[m_local_id];
    return ItemInternal::nullItem();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compare two entities.
 *
 * \retval true if they are identical (same localId())
 * \retval false otherwise
 */
inline bool
operator==(const Item& item1, const Item& item2)
{
  return item1.localId() == item2.localId();
}

/*!
 * \brief Compare two entities.
 *
 * \retval true if they are different (different localId())
 * \retval false otherwise
 */
inline bool
operator!=(const Item& item1, const Item& item2)
{
  return item1.localId() != item2.localId();
}

/*!
 * \brief Compare two entities.
 *
 * \retval true if they are less than (based on localId())
 * \retval false otherwise
 */
inline bool
operator<(const Item& item1, const Item& item2)
{
  return item1.localId() < item2.localId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemVectorView.h"
#include "arcane/core/ItemConnectedListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Node of a mesh.
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT Node
: public Item
{
  using ThatClass = Node;
  // For accessing private constructors
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index of a Node in a variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use NodeLocalId instead") Index
  : public Item::Index
  {
   public:

    typedef Item::Index Base;

   public:

    explicit Index(Int32 id)
    : Base(id)
    {}
    Index(Node item)
    : Base(item)
    {}
    operator NodeLocalId() const { return NodeLocalId{ localId() }; }
  };

 protected:

  //! Constructor reserved for enumerators
  constexpr Node(Int32 local_id, ItemSharedInfo* shared_info)
  : Item(local_id, shared_info)
  {}

 public:

  //! Type of localId()
  typedef NodeLocalId LocalIdType;

  //! Creation of a node not connected to the mesh
  Node() = default;

  //! (deprecated) Constructs a reference to the entity \a internal
  Node(ItemInternal* ainternal)
  : Item(ainternal)
  {
    ARCANE_CHECK_KIND(isNode);
  }

  //! Constructs a reference to the entity \a abase
  constexpr Node(const ItemBase& abase)
  : Item(abase)
  {
    ARCANE_CHECK_KIND(isNode);
  }

  //! Constructs a reference to the entity \a abase
  constexpr explicit Node(const Item& aitem)
  : Item(aitem)
  {
    ARCANE_CHECK_KIND(isNode);
  }

  //! Constructs a reference to the entity \a internal
  Node(const ItemInternalPtr* internals, Int32 local_id)
  : Item(internals, local_id)
  {
    ARCANE_CHECK_KIND(isNode);
  }

  //! Copy operator
  Node& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Kind of the entity
  constexpr eItemKind kind() const { return IK_Node; }

  //! Local identifier of the entity in the processor subdomain
  constexpr NodeLocalId itemLocalId() const { return NodeLocalId{ m_local_id }; }

  //! Number of edges connected to the node
  constexpr Int32 nbEdge() const { return _nbEdge(); }

  //! Number of faces connected to the node
  constexpr Int32 nbFace() const { return _nbFace(); }

  //! Number of cells connected to the node
  Int32 nbCell() const { return _nbCell(); }

  //! i-th edge of the node
  inline Edge edge(Int32 i) const;

  //! i-th face of the node
  inline Face face(Int32 i) const;

  //! i-th cell of the node
  inline Cell cell(Int32 i) const;

  //! i-th edge of the node
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! i-th face of the node
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! i-th cell of the node
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! List of edges of the node
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! List of faces of the node
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! List of cells of the node
  CellConnectedListViewType cells() const { return _cellList(); }

  //! List of edges of the node
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  //! List of faces of the node
  FaceLocalIdView faceIds() const { return _faceIds(); }

  //! List of cells of the node
  CellLocalIdView cellIds() const { return _cellIds(); }

  // AMR

  //! Enumerates the cells connected to the node
  CellVectorView _internalActiveCells(Int32Array& local_ids) const
  {
    return _toItemBase()._internalActiveCells2(local_ids);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Node* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Node* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr inline Node Item::
_node(Int32 index) const
{
  return Node(_connectivity()->nodeBase(m_local_id, index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh element based on nodes (Edge,Face,Cell).
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT ItemWithNodes
: public Item
{
  using ThatClass = ItemWithNodes;
  // For accessing private constructors
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 protected:

  //! Constructor reserved for enumerators
  constexpr ItemWithNodes(Int32 local_id, ItemSharedInfo* shared_info)
  : Item(local_id, shared_info)
  {}

 public:

  //! Creation of an entity not connected to the mesh
  ItemWithNodes() = default;

  //! (deprecated) Constructs a reference to the entity \a internal
  ItemWithNodes(ItemInternal* ainternal)
  : Item(ainternal)
  {
    ARCANE_CHECK_KIND(isItemWithNodes);
  }

  //! Constructs a reference to the entity \a abase
  constexpr ItemWithNodes(const ItemBase& abase)
  : Item(abase)
  {
    ARCANE_CHECK_KIND(isItemWithNodes);
  }

  //! Constructs a reference to the entity \a aitem
  constexpr explicit ItemWithNodes(const Item& aitem)
  : Item(aitem)
  {
    ARCANE_CHECK_KIND(isItemWithNodes);
  }

  //! Constructs a reference to the entity \a internal
  ItemWithNodes(const ItemInternalPtr* internals, Int32 local_id)
  : Item(internals, local_id)
  {
    ARCANE_CHECK_KIND(isItemWithNodes);
  }

  //! Copy operator
  ItemWithNodes& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Number of nodes of the entity
  Int32 nbNode() const { return _nbNode(); }

  //! i-th node of the entity
  Node node(Int32 i) const { return _node(i); }

  //! List of nodes of the entity
  NodeConnectedListViewType nodes() const { return _nodeList(); }

  //! List of nodes of the entity
  NodeLocalIdView nodeIds() const { return _nodeIds(); }

  //! i-th node of the entity.
  NodeLocalId nodeId(Int32 index) const { return _nodeId(index); }

  //! Number of nodes of the associated linear entity (if entity order 2 or more)
  Int32 nbLinearNode() const { return _nbLinearNode(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  ItemWithNodes* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const ItemWithNodes* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Edge of a mesh.
 *
 * Edges only exist in 3D. In 2D, you must use the 'Face' structure.
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT Edge
: public ItemWithNodes
{
  using ThatClass = Edge;
  // For accessing private constructors
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index of an Edge in a variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use EdgeLocalId instead") Index
  : public Item::Index
  {
   public:

    typedef Item::Index Base;

   public:

    explicit Index(Int32 id)
    : Base(id)
    {}
    Index(Edge item)
    : Base(item)
    {}
    operator EdgeLocalId() const { return EdgeLocalId{ localId() }; }
  };

 private:

  //! Constructor reserved for enumerators
  Edge(Int32 local_id, ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id, shared_info)
  {}

 public:

  //! Type of localId()
  typedef EdgeLocalId LocalIdType;

  //! Creates a null edge
  Edge() = default;

  //! (deprecated) Constructs a reference to the entity \a internal
  Edge(ItemInternal* ainternal)
  : ItemWithNodes(ainternal)
  {
    ARCANE_CHECK_KIND(isEdge);
  }

  //! Constructs a reference to the entity \a abase
  constexpr Edge(const ItemBase& abase)
  : ItemWithNodes(abase)
  {
    ARCANE_CHECK_KIND(isEdge);
  }

  //! Constructs a reference to the entity \a aitem
  constexpr explicit Edge(const Item& aitem)
  : ItemWithNodes(aitem)
  {
    ARCANE_CHECK_KIND(isEdge);
  }

  //! Constructs a reference to the entity \a internal
  Edge(const ItemInternalPtr* internals, Int32 local_id)
  : ItemWithNodes(internals, local_id)
  {
    ARCANE_CHECK_KIND(isEdge);
  }

  //! Copy operator
  Edge& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Kind of the entity
  constexpr eItemKind kind() const { return IK_Edge; }

  //! Local identifier of the entity in the processor subdomain
  EdgeLocalId itemLocalId() const { return EdgeLocalId{ m_local_id }; }

  //! Number of vertices of the edge
  Int32 nbNode() const { return 2; }

  //! Number of faces connected to the edge
  Int32 nbFace() const { return _nbFace(); }

  //! Number of cells connected to the edge
  Int32 nbCell() const { return _nbCell(); }

  //! i-th cell of the edge
  inline Cell cell(Int32 i) const;

  //! List of edge cells
  CellConnectedListViewType cells() const { return _cellList(); }

  //! i-th edge cell
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! List of edge cells
  CellLocalIdView cellIds() const { return _cellIds(); }

  //! i-th face of the edge
  inline Face face(Int32 i) const;

  //! List of faces of the edge
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! i-th face of the edge
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! List of faces of the edge
  FaceLocalIdView faceIds() const { return _faceIds(); }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Edge* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Edge* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr inline Edge Item::
_edge(Int32 index) const
{
  return Edge(_connectivity()->edgeBase(m_local_id, index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Face of a cell.
 *
 * \ingroup Mesh
 *
 A face is described by the ordered list of its nodes, which gives it an orientation.
 */
class ARCANE_CORE_EXPORT Face
: public ItemWithNodes
{
  using ThatClass = Face;
  // To access private constructors
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index of a Face in a variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use FaceLocalId instead") Index
  : public Item::Index
  {
   public:

    typedef Item::Index Base;

   public:

    explicit Index(Int32 id)
    : Base(id)
    {}
    Index(Face item)
    : Base(item)
    {}
    operator FaceLocalId() const { return FaceLocalId{ localId() }; }
  };

 private:

  //! Constructor reserved for enumerators
  constexpr Face(Int32 local_id, ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id, shared_info)
  {}

 public:

  //! Type of localId()
  typedef FaceLocalId LocalIdType;

  //! Creation of a face not connected to the mesh
  Face() = default;

  //! (deprecated) Constructs a reference to the \a internal entity
  Face(ItemInternal* ainternal)
  : ItemWithNodes(ainternal)
  {
    ARCANE_CHECK_KIND(isFace);
  }

  //! Constructs a reference to the \a base entity
  constexpr Face(const ItemBase& abase)
  : ItemWithNodes(abase)
  {
    ARCANE_CHECK_KIND(isFace);
  }

  //! Constructs a reference to the \a item entity
  constexpr explicit Face(const Item& aitem)
  : ItemWithNodes(aitem)
  {
    ARCANE_CHECK_KIND(isFace);
  }

  //! Constructs a reference to the \a internal entity
  Face(const ItemInternalPtr* internals, Int32 local_id)
  : ItemWithNodes(internals, local_id)
  {
    ARCANE_CHECK_KIND(isFace);
  }

  //! Copy operator
  Face& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Entity kind
  constexpr eItemKind kind() const { return IK_Face; }

  //! Local identifier of the entity in the processor subdomain
  FaceLocalId itemLocalId() const { return FaceLocalId{ m_local_id }; }

  //! Number of cells of the face (1 or 2)
  Int32 nbCell() const { return _nbCell(); }

  //! i-th cell of the face
  inline Cell cell(Int32 i) const;

  //! List of cells of the face
  CellConnectedListViewType cells() const { return _cellList(); }

  //! i-th cell of the face
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! List of cells of the face
  CellLocalIdView cellIds() const { return _cellIds(); }

  /*!
   * \brief Indicates if the face is on the subdomain boundary (i.e nbCell()==1)
   *
   * \warning A face on the subdomain boundary is not necessarily on the global mesh boundary.
   */
  bool isSubDomainBoundary() const { return (_flags() & ItemFlags::II_Boundary) != 0; }

  /*!
   * \a true if the face is on the subdomain boundary.
   * \deprecated Use isSubDomainBoundary() instead.
   */
  ARCANE_DEPRECATED_118 bool isBoundary() const { return isSubDomainBoundary(); }

  //! Indicates if the face is on the subdomain boundary facing outwards.
  bool isSubDomainBoundaryOutside() const
  {
    return isSubDomainBoundary() && (_flags() & ItemFlags::II_HasBackCell);
  }

  /*!
   * \brief Indicates if the face is on the subdomain boundary facing outwards.
   *
   * \deprecated Use isSubDomainBoundaryOutside()
   */
  ARCANE_DEPRECATED_118 bool isBoundaryOutside() const
  {
    return isSubDomainBoundaryOutside();
  }

  //! Cell associated with this boundary face (null cell if none)
  inline Cell boundaryCell() const;

  //! Cell behind the face (null cell if none)
  inline Cell backCell() const;

  //! Cell behind the face (null cell if none)
  CellLocalId backCellId() const { return CellLocalId(_toItemBase().backCellId()); }

  //! Cell in front of the face (null cell if none)
  inline Cell frontCell() const;

  //! Cell in front of the face (null cell if none)
  CellLocalId frontCellId() const { return CellLocalId(_toItemBase().frontCellId()); }

  /*!
   * \brief Opposite cell of this face to the cell \a cell.
   *
   * \pre backCell()==cell || frontCell()==cell.
   */
  inline Cell oppositeCell(Cell cell) const;

  /*!
   * \brief Opposite cell of this face to the cell \a cell.
   *
   * \pre backCell()==cell || frontCell()==cell.
   */
  CellLocalId oppositeCellId(CellLocalId cell_id) const
  {
    ARCANE_ASSERT((backCellId() == cell_id || frontCellId() == cell_id), ("cell is not connected to the face"));
    return (backCellId() == cell_id) ? frontCellId() : backCellId();
  }

  /*!
   * \brief Master face associated with this face.
   *
   * This face is non-null only if the face is tied to an interface
   * and is a slave face of that interface (i.e. isSlaveFace() is true)
   *
   * \sa ITiedInterface
   */
  Face masterFace() const { return _toItemBase().masterFace(); }

  //! \a true if it is the master face of an interface
  bool isMasterFace() const { return _toItemBase().isMasterFace(); }

  //! \a true if it is a slave face of an interface
  bool isSlaveFace() const { return _toItemBase().isSlaveFace(); }

  //! \a true if it is a slave or master face of an interface
  bool isTiedFace() const { return isSlaveFace() || isMasterFace(); }

  /*!
   * \brief List of slave faces associated with this master face.
   *
   * This list only exists for faces where isMasterFace() is true.
   * For others, it is empty.
   */
  FaceConnectedListViewType slaveFaces() const
  {
    if (_toItemBase().isMasterFace())
      return _faceList();
    return FaceConnectedListViewType();
  }

 public:

  //! Number of edges of the face
  Int32 nbEdge() const { return _nbEdge(); }

  //! i-th edge of the face
  Edge edge(Int32 i) const { return _edge(i); }

  //! List of edges of the face
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! i-th edge of the face
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! List of edges of the face
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Face* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Face* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr inline Face Item::
_face(Int32 index) const
{
  return Face(_connectivity()->faceBase(m_local_id, index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Cell of a mesh.
 *
 * \ingroup Mesh
 *
 Each cell uses memory to store its connectivity. This allows modules to
 write their loop identically regardless of the cell type. Initially, this
 is the simplest mechanism. It may be possible later to use template classes
 to process the same information statically (i.e., all connectivity is
 managed at compile time).

 Connectivity uses the <b>local</b> numbering of the cell's nodes. It is
 stored in the class variables #global_face_list for faces and
 #global_edge_list for edges.

 The connectivity used is that described in the LIMA notice version 3.1,
 with the difference that the numbering starts at zero and not at one.

 Since LIMA does not describe the pyramid, the numbering used is that of
 the degenerate hexahedron, considering that nodes 4, 5, 6, and 7 are the
 pyramid's apex.

 In the current version (1.6), edges are not taken into account globally
 (i.e.: there are no Edge entities per cell).
*/
class ARCANE_CORE_EXPORT Cell
: public ItemWithNodes
{
  using ThatClass = Cell;
  // To access private constructors
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index of a Cell in a variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use CellLocalId instead") Index
  : public Item::Index
  {
   public:

    typedef Item::Index Base;

   public:

    explicit Index(Int32 id)
    : Base(id)
    {}
    Index(Cell item)
    : Base(item)
    {}
    operator CellLocalId() const { return CellLocalId{ localId() }; }
  };

 private:

  //! Constructor reserved for enumerators
  Cell(Int32 local_id, ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id, shared_info)
  {}

 public:

  //! Type of localId()
  typedef CellLocalId LocalIdType;

  //! Constructor of a null cell
  Cell() = default;

  //! (deprecated) Constructs a reference to the \a internal entity
  Cell(ItemInternal* ainternal)
  : ItemWithNodes(ainternal)
  {
    ARCANE_CHECK_KIND(isCell);
  }

  //! Constructs a reference to the \a base entity
  constexpr Cell(const ItemBase& abase)
  : ItemWithNodes(abase)
  {
    ARCANE_CHECK_KIND(isCell);
  }

  //! Constructs a reference to the \a item entity
  constexpr explicit Cell(const Item& aitem)
  : ItemWithNodes(aitem)
  {
    ARCANE_CHECK_KIND(isCell);
  }

  //! Constructs a reference to the \a internal entity
  Cell(const ItemInternalPtr* internals, Int32 local_id)
  : ItemWithNodes(internals, local_id)
  {
    ARCANE_CHECK_KIND(isCell);
  }

  //! Copy operator
  Cell& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Entity kind
  constexpr eItemKind kind() const { return IK_Cell; }

  //! Local identifier of the entity in the processor subdomain
  CellLocalId itemLocalId() const { return CellLocalId{ m_local_id }; }

  //! Number of faces of the cell
  Int32 nbFace() const { return _nbFace(); }

  //! i-th face of the cell
  Face face(Int32 i) const { return _face(i); }

  //! List of faces of the cell
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! i-th face of the cell
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! List of faces of the cell
  FaceLocalIdView faceIds() const { return _faceIds(); }

  //! Number of edges of the cell
  Int32 nbEdge() const { return _nbEdge(); }

  //! i-th edge of the cell
  Edge edge(Int32 i) const { return _edge(i); }

  //! i-th edge of the cell
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! List of edges of the cell
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! List of edges of the cell
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  //! AMR
  //! ATT: the notion of parent is used both in the sub-mesh concept and AMR.
  //! The first AMR implementation separates the two concepts for consistency reasons.
  //! A fusion of the two notions is possible later
  //! initially, the names for AMR are in French, i.e. parent -> pere and child -> enfant
  //! a single parent
  Cell hParent() const { return Cell(_hParentBase(0)); }

  //! Number of parents for AMR
  Int32 nbHParent() const { return _nbHParent(); }

  //! Number of children for AMR
  Int32 nbHChildren() const { return _nbHChildren(); }

  //! i-th AMR child
  Cell hChild(Int32 i) const { return Cell(_hChildBase(i)); }

  //! level 0 parent for AMR
  Cell topHParent() const { return Cell(_toItemBase().topHParentBase()); }

  /*!
   * \returns \p true if the item is active (i.e. has no
   * active descendants), \p false otherwise. Note that it is sufficient to check
   * only the first child. Always returns \p true if AMR is disabled.
   */
  bool isActive() const { return _toItemBase().isActive(); }

  bool isSubactive() const { return _toItemBase().isSubactive(); }

  /*!
   * \returns \p true if the item is an ancestor (i.e. has an
   * active child or an ancestor child), \p false otherwise.
   * Always returns \p false if AMR is disabled.
   */
  bool isAncestor() const { return _toItemBase().isAncestor(); }

  /*!
   * \returns \p true if the item has children (active or not),
   * \p false otherwise. Always returns \p false if AMR is disabled.
   */
  bool hasHChildren() const { return _toItemBase().hasHChildren(); }

  /*!
   * \returns the refinement level of the current item. If the item
   * parent is \p NULL, then by convention it is at level 0,
   * otherwise it is simply at a level higher than its parent.
   */
  Int32 level() const
  {
    //! if I don't have a parent, I was created
    //! directly from a file or by the user,
    //! so I am a level 0 item
    if (this->_nbHParent() == 0)
      return 0;
    //! otherwise I am one level higher than my parent
    return (this->_hParentBase(0).level() + 1);
  }

  /*!
   * \returns the rank of the child \p (iitem).
   * example: if rank = m_internal->whichChildAmI(iitem); then
   * m_internal->hChild(rank) would be iitem;
   */
  Int32 whichChildAmI(const ItemInternal* iitem) const
  {
    return _toItemBase().whichChildAmI(iitem->localId());
  }

  /*!
   * \returns the rank of the child with \p (iitem).
   * example: if rank = m_internal->whichChildAmI(iitem); then
   * m_internal->hChild(rank) would be iitem;
   */
  Int32 whichChildAmI(CellLocalId local_id) const
  {
    return _toItemBase().whichChildAmI(local_id);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Cell* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Cell* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr inline Cell Item::
_cell(Int32 index) const
{
  return Cell(_connectivity()->cellBase(m_local_id, index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Particle.
 * \ingroup Mesh
 */
class Particle
: public Item
{
  using ThatClass = Particle;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 private:

  //! Constructor reserved for enumerators
  Particle(Int32 local_id, ItemSharedInfo* shared_info)
  : Item(local_id, shared_info)
  {}

 public:

  //! Type of localId()
  typedef ParticleLocalId LocalIdType;

  //! Constructor for a null particle
  Particle() = default;

  //! (deprecated) Constructs a reference to the \a internal entity
  Particle(ItemInternal* ainternal)
  : Item(ainternal)
  {
    ARCANE_CHECK_KIND(isParticle);
  }

  //! Constructs a reference to the \a abase entity
  constexpr Particle(const ItemBase& abase)
  : Item(abase)
  {
    ARCANE_CHECK_KIND(isParticle);
  }

  //! Constructs a reference to the \a aitem entity
  constexpr explicit Particle(const Item& aitem)
  : Item(aitem)
  {
    ARCANE_CHECK_KIND(isParticle);
  }

  //! Constructs a reference to the \a internal entity
  Particle(const ItemInternalPtr* internals, Int32 local_id)
  : Item(internals, local_id)
  {
    ARCANE_CHECK_KIND(isParticle);
  }

  //! Copy operator
  Particle& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Entity kind
  constexpr eItemKind kind() const { return IK_Particle; }

  //! Local identifier of the entity in the processor subdomain
  ParticleLocalId itemLocalId() const { return ParticleLocalId{ m_local_id }; }

  /*!
   * \brief Cell to which the particle belongs.
   * You must call setCell() before calling this function.
   * \precondition hasCell() must be true.
   */
  Cell cell() const { return _cell(0); }

  //! Cell connected to the particle
  CellLocalId cellId() const { return _cellId(0); }

  //! True if the particle is in a mesh cell
  bool hasCell() const { return (_cellId(0).localId() != NULL_ITEM_LOCAL_ID); }

  /*!
   * \brief Cell to which the particle belongs or null cell.
   * Returns cell() if the particle is in a cell or the
   * null cell if the particle is not in any cell.
   */
  Cell cellOrNull() const
  {
    Int32 cell_local_id = _cellId(0).localId();
    if (cell_local_id == NULL_ITEM_LOCAL_ID)
      return Cell();
    return _cell(0);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Particle* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Particle* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief degree of freedom class.
 *
 * \ingroup Mesh
 *
 * This new DoF item introduces a new connectivity management, offloaded
 * into properties and no longer stored in ItemSharedInfo in order to be able to create
 * new connectivities based on user needs. By default, no
 * connectivity is associated with the DoF. Necessary connectivities will be added by the user.
 */
class DoF
: public Item
{
  using ThatClass = DoF;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 private:

  //! Constructor reserved for enumerators
  constexpr DoF(Int32 local_id, ItemSharedInfo* shared_info)
  : Item(local_id, shared_info)
  {}

 public:

  using LocalIdType = DoFLocalId;

  //! Constructor for a non-connected cell
  DoF() = default;

  //! (deprecated) Constructs a reference to the \a internal entity
  DoF(ItemInternal* ainternal)
  : Item(ainternal)
  {
    ARCANE_CHECK_KIND(isDoF);
  }

  //! Constructs a reference to the \a abase entity
  constexpr DoF(const ItemBase& abase)
  : Item(abase)
  {
    ARCANE_CHECK_KIND(isDoF);
  }

  //! Constructs a reference to the \a abase entity
  constexpr explicit DoF(const Item& aitem)
  : Item(aitem)
  {
    ARCANE_CHECK_KIND(isDoF);
  }

  //! Constructs a reference to the \a internal entity
  DoF(const ItemInternalPtr* internals, Int32 local_id)
  : Item(internals, local_id)
  {
    ARCANE_CHECK_KIND(isDoF);
  }

  //! Copy operator
  DoF& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  DoF* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const DoF* operator->() const { return this; }

  //! Entity kind
  constexpr eItemKind kind() const { return IK_DoF; }

  //! Local identifier of the entity in the processor subdomain
  DoFLocalId itemLocalId() const { return DoFLocalId{ m_local_id }; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Edge Node::
edge(Int32 i) const
{
  return _edge(i);
}

inline Face Node::
face(Int32 i) const
{
  return _face(i);
}

inline Cell Node::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Face Edge::
face(Int32 i) const
{
  return _face(i);
}

inline Cell Edge::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Cell Face::
boundaryCell() const
{
  return Cell(_toItemBase().boundaryCell());
}

inline Cell Face::
backCell() const
{
  return Cell(_toItemBase().backCell());
}

inline Cell Face::
frontCell() const
{
  return Cell(_toItemBase().frontCell());
}

inline Cell Face::
oppositeCell(Cell cell) const
{
  ARCANE_ASSERT((backCell() == cell || frontCell() == cell), ("cell is not connected to the face"));
  return (backCell() == cell) ? frontCell() : backCell();
}

inline Cell Face::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemWithNodes Item::
toItemWithNodes() const
{
  ARCANE_CHECK_KIND(isItemWithNodes);
  return ItemWithNodes(*this);
}

inline Node Item::
toNode() const
{
  ARCANE_CHECK_KIND(isNode);
  return Node(*this);
}

inline Edge Item::
toEdge() const
{
  ARCANE_CHECK_KIND(isEdge);
  return Edge(*this);
}

inline Face Item::
toFace() const
{
  ARCANE_CHECK_KIND(isFace);
  return Face(*this);
}

inline Cell Item::
toCell() const
{
  ARCANE_CHECK_KIND(isCell);
  return Cell(*this);
}

inline Particle Item::
toParticle() const
{
  ARCANE_CHECK_KIND(isParticle);
  return Particle(*this);
}

inline DoF Item::
toDoF() const
{
  ARCANE_CHECK_KIND(isDoF);
  return DoF(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(Item item)
: m_local_id(item.localId())
{
}

template <typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemType item)
: ItemLocalId(item.localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Item ItemInfoListView::
operator[](ItemLocalId local_id) const
{
  return Item(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Item ItemInfoListView::
operator[](Int32 local_id) const
{
  return Item(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemType ItemInfoListViewT<ItemType>::
operator[](ItemLocalId local_id) const
{
  return ItemType(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemType ItemInfoListViewT<ItemType>::
operator[](Int32 local_id) const
{
  return ItemType(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Item ItemLocalIdToItemConverter::
operator[](ItemLocalId local_id) const
{
  return Item(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Item ItemLocalIdToItemConverter::
operator[](Int32 local_id) const
{
  return Item(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType_> inline constexpr ARCCORE_HOST_DEVICE ItemType_
ItemLocalIdToItemConverterT<ItemType_>::
operator[](ItemLocalIdType local_id) const
{
  return ItemType(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType_> inline constexpr ARCCORE_HOST_DEVICE ItemType_
ItemLocalIdToItemConverterT<ItemType_>::
operator[](Int32 local_id) const
{
  return ItemType(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemCompatibility.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
