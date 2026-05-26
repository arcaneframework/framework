// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Info about nodes in a direction X, Y, or Z of a structured mesh.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_NODEDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_NODEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypedef.h"
#include "arcane/IndexedItemConnectivityView.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Node before and after a node following a direction.
 *
 * Instances of this class are temporary and constructed via
 * NodeDirectionMng::node().
 */
class ARCANE_CARTESIANMESH_EXPORT DirNode
{
  friend NodeDirectionMng;
  friend class DirNodeLocalId;

 private:

  typedef signed char IndexType;
  static constexpr IndexType NULL_CELL = -1;

  struct DirNodeCellIndex
  {
   public:

    ARCCORE_HOST_DEVICE IndexType operator[](Int32 i) const
    {
      ARCANE_CHECK_AT(i, 8);
      return m_indexes[i];
    }

   public:

    IndexType m_indexes[8];
  };

 private:

  // Only NodeDirectionMng has the right to construct a DirNode.
  DirNode(Node current, Node next, Node prev, DirNodeCellIndex idx)
  : m_current(current)
  , m_previous(prev)
  , m_next(next)
  , m_cell_index(idx)
  {}

 public:

  //! Previous cell
  Node previous() const { return m_previous; }
  //! Previous cell
  NodeLocalId previousId() const { return m_previous.itemLocalId(); }
  //! Next cell
  Node next() const { return m_next; }
  //! Next cell
  NodeLocalId nextId() const { return m_next.itemLocalId(); }
  /*!
   * \brief Index in the list of cells for this node in a
   * cell based on its position.
   *
   * Possible values for \a position are given by the enumeration
   * eCellNodePosition.
   */
  Int32 cellIndex(Int32 position) const { return m_cell_index[position]; }
  /*!
   * \brief Local index of a cell based on its position relative to this node.
   *
   * Possible values for \a position are given by the enumeration
   * eCellNodePosition.
   */
  CellLocalId cellId(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? CellLocalId(NULL_ITEM_LOCAL_ID) : CellLocalId(m_current.cellId(x));
  }
  /*!
   * \brief Cell based on its position relative to this node.
   *
   * Possible values for \a position are given by the enumeration
   * eCellNodePosition.
   */
  Cell cell(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? Cell() : Cell(m_current.cell(x));
  }

  //! NextLeftCell: Cell in front and to the left in the direction
  Cell nextLeftCell() const { return cell(CNP_NextLeft); }
  //! NextRightCell: Cell in front and to the right in the direction
  Cell nextRightCell() const { return cell(CNP_NextRight); }
  //! PreviousRightCell: Cell behind and to the right in the direction
  Cell previousRightCell() const { return cell(CNP_PreviousRight); }
  //! PreviousLeftCell: Cell behind and to the left in the direction
  Cell previousLeftCell() const { return cell(CNP_PreviousLeft); }

  //! NextLeftCell: Cell in front and to the left in the direction
  CellLocalId nextLeftCellId() const { return cellId(CNP_NextLeft); }
  //! NextRightCell: Cell in front and to the right in the direction
  CellLocalId nextRightCellId() const { return cellId(CNP_NextRight); }
  //! PreviousRightCell: Cell behind and to the right in the direction
  CellLocalId previousRightCellId() const { return cellId(CNP_PreviousRight); }
  //! PreviousLeftCell: Cell behind and to the left in the direction
  CellLocalId previousLeftCellId() const { return cellId(CNP_PreviousLeft); }

  //! TopNextLeftCell: Cell in front and to the left in the direction
  Cell topNextLeftCell() const { return cell(CNP_TopNextLeft); }
  //! TopNextRightCell: Cell in front and to the right in the direction
  Cell topNextRightCell() const { return cell(CNP_TopNextRight); }
  //! TopPreviousRightCell: Cell behind and to the right in the direction
  Cell topPreviousRightCell() const { return cell(CNP_TopPreviousRight); }
  //! TopPreviousLeftCell: Cell behind and to the left in the direction
  Cell topPreviousLeftCell() const { return cell(CNP_TopPreviousLeft); }

  //! TopNextLeftCell: Cell in front and to the left in the direction
  CellLocalId topNextLeftCellId() const { return cellId(CNP_TopNextLeft); }
  //! TopNextRightCell: Cell in front and to the right in the direction
  CellLocalId topNextRightCellId() const { return cellId(CNP_TopNextRight); }
  //! TopPreviousRightCell: Cell behind and to the right in the direction
  CellLocalId topPreviousRightCellId() const { return cellId(CNP_TopPreviousRight); }
  //! TopPreviousLeftCell: Cell behind and to the left in the direction
  CellLocalId topPreviousLeftCellId() const { return cellId(CNP_TopPreviousLeft); }

 private:

  Node m_current;
  Node m_previous;
  Node m_next;
  DirNodeCellIndex m_cell_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Node before and after a node following a direction.
 *
 * Instances of this class are temporary and constructed via
 * NodeDirectionMng::dirNodeId().
 */
class ARCANE_CARTESIANMESH_EXPORT DirNodeLocalId
{
  friend NodeDirectionMng;

 private:

  typedef signed char IndexType;
  static constexpr IndexType NULL_CELL = -1;

 private:

  // Only NodeDirectionMng has the right to construct a DirNode.
  ARCCORE_HOST_DEVICE DirNodeLocalId(NodeLocalId current, NodeLocalId next, NodeLocalId prev,
                                     DirNode::DirNodeCellIndex idx,
                                     IndexedNodeCellConnectivityView view)
  : m_current(current)
  , m_previous(prev)
  , m_next(next)
  , m_cell_index(idx)
  , m_view(view)
  {}

 public:

  //! Previous cell
  ARCCORE_HOST_DEVICE NodeLocalId previous() const { return m_previous; }
  //! Previous cell
  ARCCORE_HOST_DEVICE NodeLocalId previousId() const { return m_previous; }
  //! Next cell
  ARCCORE_HOST_DEVICE NodeLocalId next() const { return m_next; }
  //! Next cell
  ARCCORE_HOST_DEVICE NodeLocalId nextId() const { return m_next; }
  /*!
   * \brief Index in the list of cells for this node in a
   * cell based on its position.
   *
   * Possible values for \a position are given by the enumeration
   * eCellNodePosition.
   */
  ARCCORE_HOST_DEVICE Int32 cellIndex(Int32 position) const { return m_cell_index[position]; }
  /*!
   * \brief Local index of a cell based on its position relative to this node.
   *
   * Possible values for \a position are given by the enumeration
   * eCellNodePosition.
   */
  ARCCORE_HOST_DEVICE CellLocalId cellId(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? CellLocalId(NULL_ITEM_LOCAL_ID) : m_view.cellId(m_current, x);
  }

  //! NextLeftCell: Cell in front and to the left in the direction
  ARCCORE_HOST_DEVICE CellLocalId nextLeftCellId() const { return cellId(CNP_NextLeft); }
  //! NextRightCell: Cell in front and to the right in the direction
  ARCCORE_HOST_DEVICE CellLocalId nextRightCellId() const { return cellId(CNP_NextRight); }
  //! PreviousRightCell: Cell behind and to the right in the direction
  ARCCORE_HOST_DEVICE CellLocalId previousRightCellId() const { return cellId(CNP_PreviousRight); }
  //! PreviousLeftCell: Cell behind and to the left in the direction
  ARCCORE_HOST_DEVICE CellLocalId previousLeftCellId() const { return cellId(CNP_PreviousLeft); }

  //! TopNextLeftCell: Cell in front and to the left in the direction
  ARCCORE_HOST_DEVICE CellLocalId topNextLeftCellId() const { return cellId(CNP_TopNextLeft); }
  //! TopNextRightCell: Cell in front and to the right in the direction
  ARCCORE_HOST_DEVICE CellLocalId topNextRightCellId() const { return cellId(CNP_TopNextRight); }
  //! TopPreviousRightCell: Cell behind and to the right in the direction
  ARCCORE_HOST_DEVICE CellLocalId topPreviousRightCellId() const { return cellId(CNP_TopPreviousRight); }
  //! TopPreviousLeftCell: Cell behind and to the left in the direction
  ARCCORE_HOST_DEVICE CellLocalId topPreviousLeftCellId() const { return cellId(CNP_TopPreviousLeft); }

 private:

  NodeLocalId m_current;
  NodeLocalId m_previous;
  NodeLocalId m_next;
  DirNode::DirNodeCellIndex m_cell_index;
  IndexedNodeCellConnectivityView m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Info about nodes in a specific direction X, Y, or Z
 * of a structured mesh.
 */
class ARCANE_CARTESIANMESH_EXPORT NodeDirectionMng
{
  friend CartesianMeshImpl;
  friend CartesianMeshPatch;
  class Impl;
  using IndexType = DirNode::IndexType;
  using DirNodeCellIndex = DirNode::DirNodeCellIndex;

 private:

  struct ItemDirectionInfo
  {
   public:

    /*!
     * \brief Default constructor.
     * \warning The values m_next_item and m_previous_item are initialized
     * to nullptr.
     */
    ItemDirectionInfo() = default;
    ItemDirectionInfo(Int32 next_lid, Int32 prev_lid)
    : m_next_lid(next_lid)
    , m_previous_lid(prev_lid)
    {}

   public:

    //! Entity after the current entity in the direction
    NodeLocalId m_next_lid;
    //! Entity before the current entity in the direction
    NodeLocalId m_previous_lid;

   public:

    void setCellIndexes(IndexType idx[8])
    {
      for (int i = 0; i < 8; ++i)
        m_cell_index.m_indexes[i] = idx[i];
    }
    DirNodeCellIndex m_cell_index;
  };

 public:

  /*!
   * \brief Creates an empty instance.
   *
   * The instance is not valid until _internalInit() has been called.
   */
  NodeDirectionMng();

 public:

  //! Direction node corresponding to node \a n
  DirNode node(Node n) const
  {
    return _node(n.localId());
  }

  //! Direction node corresponding to node \a n
  DirNode node(NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Direction node corresponding to node \a n
  DirNode dirNode(NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Direction node ID corresponding to node \a n
  ARCCORE_HOST_DEVICE DirNodeLocalId dirNodeId(NodeLocalId n) const
  {
    return _dirNodeId(n);
  }

  //! Group of all nodes in the direction.
  NodeGroup allNodes() const;

  /*!
   * \brief Group of all overlap nodes in the direction.
   *
   * An overlap node is a node that only has overlap cells around it.
   *
   *   0   1  2  3  4
   * ┌───┬──┬──┬──┬──┐
   * │   │  │  │  │  │
   * │   ├──┼──┼──┼──┤
   * │   │  │  │  │  │
   * └───┴──┴──┴──┴──┘
   *
   * 0 : level -1
   * 1 and 2 : Overlap cells (overlapCells)
   * 3 : Outer cells (outerCells)
   * 4 : Inner cells (innerCells)
   *
   * The overlap cell layer designates the layer of cells of the same
   * level around the patch. These cells may belong to one or more
   * patches.
   */
  NodeGroup overlapNodes() const;

  /*!
   * \brief Group of all patch nodes in the direction.
   *
   * Patch nodes are nodes that do not have all their cells as overlap cells. TODO reformulate
   * (`innerNodes() + outerNodes()` or simply `!overlapNodes()`)
   *
   * \warning Nodes at the domain boundary (i.e., having only "outer" cells) are included in this group.
   */
  NodeGroup inPatchNodes() const;

  /*!
   * \brief Group of all inner nodes in the direction.
   *
   * A node is considered inner if its previous or next node is not null.
   */
  NodeGroup innerNodes() const;

  /*!
   * \brief Group of all outer nodes in the direction.
   *
   * A node is considered outer if its previous or next node is overlap or is null (if it is at the
   * domain boundary or if there are no overlap cell layers).
   *
   * \note Nodes between patches are not duplicated. Therefore, some nodes
   * in this group may also be in an outerNodes() of another patch.
   */
  NodeGroup outerNodes() const;

  //! Direction node corresponding to node \a n.
  DirNode operator[](Node n)
  {
    return _node(n.localId());
  }

  //! Direction node corresponding to node \a n.
  DirNode operator[](NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Direction node corresponding to the node iterator \a inode.
  DirNode operator[](NodeEnumerator inode) const
  {
    return _node(inode.itemLocalId());
  }

  //! Direction value
  eMeshDirection direction() const
  {
    return m_direction;
  }

 protected:

  /*!
   * \internal
   * \brief Calculates the information about nodes associated with cells of
   * the direction \a cell_dm. \a all_nodes is the group of all nodes of the cells
   * managed by \a cell_dm.
   * Assumes init() has been called.
   */
  void _internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes,
                             const VariableCellReal3& cells_center);

  void _internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes);

  /*!
   * \internal
   * Initializes the instance.
   */
  void _internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index);

  /*!
   * \internal
   * Destroys the resources associated with the instance.
   */
  void _internalDestroy();

  /*!
   * \brief Resizes the container holding the \a ItemDirectionInfo.
   *
   * This invalidates current instances of NodeDirectionMng.
   */
  void _internalResizeInfos(Int32 new_size);

 private:

  SmallSpan<ItemDirectionInfo> m_infos_view;
  NodeInfoListView m_nodes;
  eMeshDirection m_direction;
  IndexedNodeCellConnectivityView m_node_cell_view;
  Impl* m_p;

 private:

  //! Direction node corresponding to local node number \a local_id
  DirNode _node(Int32 local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirNode(m_nodes[local_id], m_nodes[d.m_next_lid], m_nodes[d.m_previous_lid], d.m_cell_index);
  }

  //! Direction node ID corresponding to local node number \a local_id
  ARCCORE_HOST_DEVICE DirNodeLocalId _dirNodeId(NodeLocalId local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id.localId()];
    return DirNodeLocalId(local_id, d.m_next_lid, d.m_previous_lid, d.m_cell_index, m_node_cell_view);
  }

  void _computeNodeCellInfos(const CellDirectionMng& cell_dm,
                             const VariableCellReal3& cells_center);
  void _computeNodeCellInfos() const;
  void _filterNodes();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
