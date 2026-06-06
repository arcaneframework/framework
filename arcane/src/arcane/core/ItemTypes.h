// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypes.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Declaration of types related to mesh entities.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPES_H
#define ARCANE_CORE_ITEMTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ItemTypes.h
 *
 * \brief Declarations of types on entities.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// Define whether to add an offset to the 'ItemVector' and
// 'ItemEnumerator' classes. This changes the size of these structures and
// therefore, user code must be recompiled if this '#define' is changed.
#define ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW

// Define whether to hide access methods to the internal structures
// of connectivities. For now (mid-2023), this is only done for internal
// Arcane sources, but later it will need to be generalized.
// (The macro ARCANE_FORCE_... is defined in the main CMakeLists.txt)
#ifdef ARCANE_FORCE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
#define ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
#endif

// Define whether to use specific classes to manage
// connected entities (otherwise, ItemVectorView is used)
#define ARCANE_USE_SPECIFIC_ITEMCONNECTED

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Node;
class Cell;
class Edge;
class Face;
class Particle;
class DoF;

class Item;
class ItemWithNodes;
class ItemInternal;
class ItemBase;
class MutableItemBase;
namespace impl
{
using ItemBase = ::Arcane::ItemBase;
using MutableItemBase = ::Arcane::MutableItemBase;
template<int Extent = DynExtent> class ItemIndexedListView;
class ItemLocalIdListContainerView;
}


class IMesh;
class IPrimaryMesh;
class IItemFamily;
class IParticleFamily;
class IDoFFamily;

class ItemLocalId;
class ItemTypeId;
template<typename T> class ItemLocalIdT;
using NodeLocalId = ItemLocalIdT<Node>;
using EdgeLocalId = ItemLocalIdT<Edge>;
using FaceLocalId = ItemLocalIdT<Face>;
using CellLocalId = ItemLocalIdT<Cell>;
using ParticleLocalId = ItemLocalIdT<Particle>;
using DoFLocalId = ItemLocalIdT<DoF>;
class IndexedItemConnectivityViewBase;
class IndexedItemConnectivityViewBase2;
class IndexedItemConnectivityAccessor;
class ItemInternalConnectivityList;
class ItemInternalVectorView;
class ItemIndexArrayView;
class ItemLocalIdListView;

template<typename T> class ItemLocalIdListViewT;
template<typename ItemType> using ItemLocalIdViewT ARCANE_DEPRECATED_REASON("Use 'ItemLocalIdListView' type instead") = ItemLocalIdListViewT<ItemType>;

class ItemGroup;
class ItemGroupImpl;
template<typename T> class ItemGroupT;

class ItemPairGroup;
template<typename ItemKind,typename SubItemKind> class ItemPairGroupT;

class ItemVector;
template<typename T> class ItemVectorT;

class ItemVectorViewConstIterator;
template<typename ItemType>
class ItemVectorViewConstIteratorT;

class ItemConnectedListViewConstIterator;
template<typename ItemType>
class ItemConnectedListViewConstIteratorT;

// (April 2022) Creates a typedef of 'ItemLocalIdViewT' to 'ItemLocalIdView'
// for compatibility with existing code. To be removed as soon as possible.
template <typename ItemType>
using ItemLocalIdView ARCANE_DEPRECATED_REASON("Use 'ItemLocalIdViewT' instead") = ItemLocalIdListViewT<ItemType>;

template<typename ItemType1,typename ItemType2>
class IndexedItemConnectivityViewT;

// (April 2022) Creates a typedef of 'IndexedItemConnectivityView' to 'IndexedItemConnectivityViewT'
// for compatibility with existing code. To be removed as soon as possible.
template<typename ItemType1,typename ItemType2>
using IndexedItemConnectivityView ARCANE_DEPRECATED_REASON("Use 'IndexedItemConnectivityViewT' instead") = IndexedItemConnectivityViewT<ItemType1,ItemType2>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Group of nodes connected to nodes
typedef ItemPairGroupT<Node,Node> NodeNodeGroup;
//! Group of edges connected to nodes
typedef ItemPairGroupT<Edge,Node> EdgeNodeGroup;
//! Group of faces connected to nodes
typedef ItemPairGroupT<Face,Node> FaceNodeGroup;
//! Group of cells connected to nodes
typedef ItemPairGroupT<Cell,Node> CellNodeGroup;

//! Group of nodes connected to faces
typedef ItemPairGroupT<Node,Face> NodeFaceGroup;
//! Group of edges connected to faces
typedef ItemPairGroupT<Edge,Face> EdgeFaceGroup;
//! Group of faces connected to faces
typedef ItemPairGroupT<Face,Face> FaceFaceGroup;
//! Group of cells connected to faces
typedef ItemPairGroupT<Cell,Face> CellFaceGroup;

//! Group of nodes connected to cells
typedef ItemPairGroupT<Node,Cell> NodeCellGroup;
//! Group of edges connected to cells
typedef ItemPairGroupT<Edge,Cell> EdgeCellGroup;
//! Group of faces connected to cells
typedef ItemPairGroupT<Face,Cell> FaceCellGroup;
//! Group of cells connected to cells
typedef ItemPairGroupT<Cell,Cell> CellCellGroup;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Group of nodes.
 */
typedef ItemGroupT<Node> NodeGroup;
/*!
 * \ingroup Mesh
 * \brief Group of edges
 * \note This class is not implemented.
 */
typedef ItemGroupT<Edge> EdgeGroup;
/*!
 * \ingroup Mesh
 * \brief Group of faces.
 */
typedef ItemGroupT<Face> FaceGroup;
/*!
 * \ingroup Mesh
 * \brief Group of cells.
 */
typedef ItemGroupT<Cell> CellGroup;
/*!
 * \ingroup Mesh
 * \brief Group of particles.
 */
typedef ItemGroupT<Particle> ParticleGroup;
/*!
 * \ingroup Mesh
 * \brief Group of Degrees of Freedom.
 */
typedef ItemGroupT<DoF> DoFGroup;
/*!
 * \ingroup Mesh
 * \internal
 * \brief Enumerator over the internal part of an entity.
 */
class ItemInternalEnumerator;
class ItemEnumerator;
template<typename ItemType>
class ItemEnumeratorT;

class ItemConnectedEnumerator;
template<typename ItemType>
class ItemConnectedEnumeratorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EnumeratorTraceInfo;
class SimdItemEnumeratorBase;

class ItemEnumeratorBase;
template<typename ItemType>
class ItemEnumeratorBaseT;

template<typename ItemType>
class ItemConnectedEnumeratorBaseT;

template<typename ItemType>
class SimdItemEnumeratorT;
template<typename ItemType>
class SimdItemT;

/*!
 * \ingroup Mesh
 * \brief View over an entity vector.
 */
class ItemVectorView;
template<typename ItemType>
class ItemVectorViewT;

/*!
 * \ingroup Mesh
 * \brief View over a connectivity list
 */
template<int Extent = DynExtent> class ItemConnectedListView;
template<typename ItemType, int Extent = DynExtent>
class ItemConnectedListViewT;

/*!
 * \ingroup Mesh
 * \brief Enumerator over an entity pair.
 */
class ItemPairEnumerator;
template<typename ItemType,typename SubItemType>
class ItemPairEnumeratorT;

/*!
 * \ingroup Mesh
 * \brief Enumerators over nodes.
 */
typedef ItemEnumeratorT<Node> NodeEnumerator;
/*!
 * \ingroup Mesh
 * \brief Enumerators over edges
 */
typedef ItemEnumeratorT<Edge> EdgeEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumerators over faces.
 */
typedef ItemEnumeratorT<Face> FaceEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumerators over cells.
 */
typedef ItemEnumeratorT<Cell> CellEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumerators over particles.
 */
typedef ItemEnumeratorT<Particle> ParticleEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumerators over DoFs.
 */
typedef ItemEnumeratorT<DoF> DoFEnumerator;

/*!
 * \ingroup Mesh
 * \brief View over a vector of nodes.
 */
typedef ItemVectorViewT<Node> NodeVectorView;
/*!
 * \ingroup Mesh
 * \brief View over a vector of edges.
 */
typedef ItemVectorViewT<Edge> EdgeVectorView;
/*!
 * \ingroup Mesh
 * \brief View over a vector of faces.
 */
typedef ItemVectorViewT<Face> FaceVectorView;
/*!
 * \ingroup Mesh
 * \brief View over a vector of cells.
 */
typedef ItemVectorViewT<Cell> CellVectorView;
/*!
 * \ingroup Mesh
 * \brief View over a vector of particles.
 */
typedef ItemVectorViewT<Particle> ParticleVectorView;

/*!
 * \ingroup Mesh
 * \brief View over a vector of degrees of freedom.
 */
typedef ItemVectorViewT<DoF> DoFVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief View over a list of nodes connected to an entity
 */
using NodeConnectedListView = ItemConnectedListViewT<Node>;
/*!
 * \ingroup Mesh
 * \brief View over a list of edges connected to an entity
 */
using EdgeConnectedListView = ItemConnectedListViewT<Edge>;
/*!
 * \ingroup Mesh
 * \brief View over a list of faces connected to an entity
 */
using FaceConnectedListView = ItemConnectedListViewT<Face>;
/*!
 * \ingroup Mesh
 * \brief View over a list of cells connected to an entity
 */
using CellConnectedListView = ItemConnectedListViewT<Cell>;
/*!
 * \ingroup Mesh
 * \brief View over a list of DoFs connected to an entity
 */
using DoFConnectedListView = ItemConnectedListViewT<DoF>;

#ifdef ARCANE_USE_SPECIFIC_ITEMCONNECTED
//! List of connected entities
using ItemConnectedListViewType = ItemConnectedListView<DynExtent>;
//! List of connected nodes
using NodeConnectedListViewType = NodeConnectedListView;
//! List of connected edges
using EdgeConnectedListViewType = EdgeConnectedListView;
//! List of connected faces
using FaceConnectedListViewType = FaceConnectedListView;
//! List of connected cells
using CellConnectedListViewType = CellConnectedListView;
//! Generic list of connected entities
template<typename ItemType> using ItemConnectedListViewTypeT = ItemConnectedListViewT<ItemType>;
#else
//! List of connected entities
using ItemConnectedListViewType = ItemVectorView;
//! List of connected nodes
using NodeConnectedListViewType = NodeVectorView;
//! List of connected edges
using EdgeConnectedListViewType = EdgeVectorView;
//! List of connected faces
using FaceConnectedListViewType = FaceVectorView;
//! List of connected cells
using CellConnectedListViewType = CellVectorView;
//! Generic list of connected entities
template<typename ItemType> using ItemConnectedListViewTypeT = ItemVectorViewT<ItemType>;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Collection of node groups. */
typedef Collection<NodeGroup> NodeGroupCollection;
/*! \brief Collection of edge groups. */
typedef Collection<EdgeGroup> EdgeGroupCollection;
/*! \brief Collection of face groups. */
typedef Collection<FaceGroup> FaceGroupCollection;
/*! \brief Collection of cell groups. */
typedef Collection<CellGroup> CellGroupCollection;
/*! \brief Collection of particle groups. */
typedef Collection<ParticleGroup> ParticleGroupCollection;
/*! \brief Collection of degrees of freedom groups. */
typedef Collection<DoFGroup> DoFGroupCollection;


/*! \brief Array of node groups. */
typedef List<NodeGroup> NodeGroupList;
/*! \brief Array of edge groups. */
typedef List<EdgeGroup> EdgeGroupList;
/*! \brief Array of face groups. */
typedef List<FaceGroup> FaceGroupList;
/*! \brief Array of cell groups. */
typedef List<CellGroup> CellGroupList;
/*! \brief Array of particle groups. */
typedef List<ParticleGroup> ParticleGroupList;
/*! \brief Array of degrees of freedom groups. */
typedef List<DoFGroup> DoFGroupList;

/*!
 * \ingroup Mesh
 * \brief View over the localId() of a list of nodes.
 */
typedef ItemLocalIdListViewT<Node> NodeLocalIdView;
/*!
 * \ingroup Mesh
 * \brief View over the localId() of a list of nodes.
 */
using NodeLocalIdListView = ItemLocalIdListViewT<Node>;
/*!
 * \ingroup Mesh
 * \brief View over the localId() of a list of edges.
 */
typedef ItemLocalIdListViewT<Edge> EdgeLocalIdView;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of edges.
 */
using EdgeLocalIdListView = ItemLocalIdListViewT<Edge>;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of faces.
 */
typedef ItemLocalIdListViewT<Face> FaceLocalIdView;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of faces.
 */
using FaceLocalIdListView = ItemLocalIdListViewT<Face>;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of cells.
 */
typedef ItemLocalIdListViewT<Cell> CellLocalIdView;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of cells.
 */
using CellLocalIdListView = ItemLocalIdListViewT<Cell>;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of particles.
 */
typedef ItemLocalIdListViewT<Particle> ParticleLocalIdView;
/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of particles.
 */
using ParticleLocalIdListView = ItemLocalIdListViewT<Particle>;

/*!
 * \ingroup Mesh
 * \brief View on the localIds() of a list of DoFs.
 */
typedef ItemLocalIdListViewT<DoF> DoFLocalIdView;
using DoFLocalIdListView = ItemLocalIdListViewT<DoF>;

/*! \brief Type of the internal list of entities
  
  \deprecated Use ItemInternalArrayView.
  
*/
typedef ConstArrayView<ItemInternal*> ItemInternalList;

typedef ConstArrayView<ItemInternal*> ItemInternalArrayView;

typedef ArrayView<ItemInternal*> ItemInternalMutableArrayView;

class IItemOperationByBasicType;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInfoListView;
template <typename ItemType> class ItemInfoListViewT;
class NodeInfoListView;
class EdgeInfoListView;
class FaceInfoListView;
class CellInfoListView;
class ParticleInfoListView;
class DoFInfoListView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemEnumeratorIndex;
template<typename ItemType>
class ItemEnumeratorIndexT;
//! Index of an enumeration on nodes
using NodeEnumeratorIndex = ItemEnumeratorIndexT<Node>;
//! Index of an enumeration on edges
using EdgeEnumeratorIndex = ItemEnumeratorIndexT<Edge>;
//! Index of an enumeration on faces
using FaceEnumeratorIndex = ItemEnumeratorIndexT<Face>;
//! Index of an enumeration on cells
using CellEnumeratorIndex = ItemEnumeratorIndexT<Cell>;
//! Index of an enumeration on particles
using ParticleEnumeratorIndex = ItemEnumeratorIndexT<Particle>;
//! Index of an enumeration on DoFs
using DoFEnumeratorIndex = ItemEnumeratorIndexT<DoF>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Connectivity usage policy.
 *
 * This enumeration serves to transition between historical connectivities
 * and the new implementation.
 *
 * Currently, only the InternalConnectivityPolicy::NewOnly value is used
 */
enum class InternalConnectivityPolicy
{
  /*!
   * \brief Historical connectivities.
   *
   * This mode is identical to the mode before the incorporation of new
   * connectivities. Its memory footprint is the smallest of all available modes.
   * \warning This mode is no longer operational.
   */
  Legacy,
  /*!
   * \brief Uses historical connectivities and allocates accessors for
   * these connectivities
   * \warning This mode is no longer operational.
   */
  LegacyAndAllocAccessor,
  /*!
   * \brief Allocates old and new connectivities
   * and uses the old ones via new accessors in ItemInternal.
   * \warning This mode is no longer operational.
   */
  LegacyAndNew,
  /*!
   * \brief Allocates old and new connectivities
   * and uses the new ones via new accessors in ItemInternal.
   * \warning This mode is no longer operational.
   */
  NewAndLegacy,
  /*!
   * \brief Allocates old and new connectivities
   * uses the new ones via new accessors in ItemInternal
   * and relies on a dependency graph of families (Families,Connectivities).
   * \warning This mode is no longer operational.
   */
  NewWithDependenciesAndLegacy,
  /*!
   * \brief Allocates only the new connectivities
   */
  NewOnly,
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Static class for obtaining information about the
 * connectivity configuration.
 */
class ARCANE_CORE_EXPORT InternalConnectivityInfo
{
 public:
  //! True if legacy connectivities are active
  static constexpr bool hasLegacyConnectivity(InternalConnectivityPolicy) { return false; }
  //! True if new connectivities are active
  static constexpr bool hasNewConnectivity(InternalConnectivityPolicy) { return true; }
  /*!
   * \brief Indicates whether new connectivities are used to access
   * entities in ItemInternal.
   */
  static constexpr bool useNewConnectivityAccessor(InternalConnectivityPolicy) { return true; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Vector of nodes.
 */
typedef ItemVectorT<Node> NodeVector;
/*!
 * \ingroup Mesh
 * \brief Vector of edges
 * \note This class is not implemented.
 */
typedef ItemVectorT<Edge> EdgeVector;
/*!
 * \ingroup Mesh
 * \brief Vector of faces.
 */
typedef ItemVectorT<Face> FaceVector;
/*!
 * \ingroup Mesh
 * \brief Vector of cells.
 */
typedef ItemVectorT<Cell> CellVector;
/*!
 * \ingroup Mesh
 * \brief Vector of particles.
 */
typedef ItemVectorT<Particle> ParticleVector;
/*!
 * \ingroup Mesh
 * \brief Vector of degrees of freedom.
 */
typedef ItemVectorT<DoF> DoFVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh elements.
 *
 * To be specialized by element type.
 */
template<class T>
class ItemTraitsT
{
 public:
   //! Entity kind
  static eItemKind kind() { return IK_Unknown; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c Node.
 */
template<>
class ItemTraitsT<Node>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<Node> ItemTraitsType;
  //! Type of the mesh entity
  typedef Node ItemType;
  //! Type of the entity group
  typedef NodeGroup ItemGroupType;
  //! Type of the localId()
  typedef NodeLocalId LocalIdType;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_Node; }

  //! Name of the associated default family
  static const char* defaultFamilyName() { return "Node"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c Edge.
 */
template<>
class ItemTraitsT<Edge>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<Edge> ItemTraitsType;
  //! Type of the mesh entity
  typedef Edge ItemType;
  //! Type of the entity group
  typedef EdgeGroup ItemGroupType;
  //! Type of the localId()
  typedef EdgeLocalId LocalIdType;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_Edge; }

  //! Name of the associated default family
  static const char* defaultFamilyName() { return "Edge"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c Face.
 */
template<>
class ItemTraitsT<Face>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<Face> ItemTraitsType;
  //! Type of the mesh entity
  typedef Face ItemType;
  //! Type of the entity group
  typedef FaceGroup ItemGroupType;
  //! Type of the localId()
  typedef FaceLocalId LocalIdType;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_Face; }

  //! Name of the associated default family
  static const char* defaultFamilyName() { return "Face"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c Cell.
 */
template<>
class ItemTraitsT<Cell>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<Cell> ItemTraitsType;
  //! Type of the mesh entity
  typedef Cell ItemType;
  //! Type of the entity group
  typedef CellGroup ItemGroupType;
  //! Type of the localId()
  typedef CellLocalId LocalIdType;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_Cell; }

  //! Name of the associated default family
  static const char* defaultFamilyName() { return "Cell"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c Particle.
 */
template<>
class ItemTraitsT<Particle>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<Particle> ItemTraitsType;
  //! Type of the mesh entity
  typedef Particle ItemType;
  //! Type of the entity group
  typedef ParticleGroup ItemGroupType;
  //! Type of the localId()
  typedef ParticleLocalId LocalIdType;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_Particle; }

  //! Name of the associated default family
  static const char* defaultFamilyName() { return nullptr; }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Characteristics of mesh entities of type \c DoF
 */
template<>
class ItemTraitsT<DoF>
{
 public:

  //! Type of this class
  typedef ItemTraitsT<DoF> ItemTraitsType;
  //! Type of the mesh entity
  typedef DoF ItemType;
  //! Type of the entity group
  typedef DoFGroup ItemGroupType;
  //! Type of the localId()
  using LocalIdType = ItemLocalIdT<DoF>;

 public:

  //! Entity kind
  static eItemKind kind() { return IK_DoF; }

  // NOTE: GG: should be nullptr because there is no default for the DoF family?
  //! Name of the associated default family
  static const char* defaultFamilyName() { return nullptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Characteristics associated with 'localId()'.
 */
template<typename ItemType>
class ItemLocalIdTraitsT
{
 public:

  //! Type of the localId()
  using LocalIdType = typename ItemTraitsT<ItemType>::LocalIdType;
};

//! Specialization for 'Item' which does not have 'ItemTraitsT'.
template<>
class ItemLocalIdTraitsT<Item>
{
 public:
  //! Type of the localId()
  using LocalIdType = ItemLocalId;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
