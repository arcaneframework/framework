// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianConnectivity.h                                     (C) 2000-2025 */
/*                                                                           */
/* Connectivity information of a Cartesian mesh.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANCONNECTIVITY_H
#define ARCANE_CARTESIANMESH_CARTESIANCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/Item.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Connectivity information of a Cartesian mesh.
 *
 * Like all objects related to the Cartesian mesh, these instances are only
 * valid as long as the mesh topology does not change.
 *
 * This class serves for both 2D connectivities and 3D connectivities. Methods
 * starting with topZ are only valid in 3D.
 *
 * The method names follow the following nomenclature:
 * - topZ/.: for the Z direction
 * - upper/lower: for the Y direction
 * - left/right: for the X direction
 *
 * For the connectivity of nodes around a coordinate cell (X0,Y0,Z0),
 * the coordinate node (X,Y,Z) is retrieved as follows:
 * - In 3D, topZ if Z>Z0, otherwise no prefix. In 2D, never a prefix.
 * - upper if Y>Y0, lower otherwise,
 * - right if X>X0, left otherwise,
 *
 * So for example, if Z>Z0, Y<Y0 and X>X0, the method name is topZLowerRight().
 * If Z<Z0, Y>Y0 and X>X0, the name is upperRight().
 *
 * The functionality is the same for the connectivities of cells around a node.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianConnectivity
{
  // NOTE: For now, we must keep the connected entities by
  // direction, because the numbering does not allow them to be easily retrieved.
  // Eventually, we will be able to deduce it directly.

  friend class CartesianConnectivityLocalId;
  friend class CartesianMeshImpl;

  /*!
   * \brief Enumerated type indicating the position.
   * \warning The exact values should not be used because they are
   * susceptible to change.
   */
  enum ePosition
  {
    P_UpperLeft = 0,
    P_UpperRight = 1,
    P_LowerRight = 2,
    P_LowerLeft = 3,

    P_TopZUpperLeft = 4,
    P_TopZUpperRight = 5,
    P_TopZLowerRight = 6,
    P_TopZLowerLeft = 7
  };

 private:

  //! List of the 8 entities around another entity
  struct Index
  {
    friend class CartesianConnectivity;

   private:

    void fill(Int32 i) { v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = v[6] = v[7] = i; }
    Int32 v[8];
  };

  //! Permutation in Index for each direction
  struct Permutation
  {
    friend class CartesianConnectivity;

   public:

    void compute();

   private:

    ePosition permutation[3][8];
  };

 public:

  //! Cell top left of node \a n
  Cell upperLeft(Node n) const { return _nodeToCell(n, P_UpperLeft); }
  //! Cell top right of node \a n
  Cell upperRight(Node n) const { return _nodeToCell(n, P_UpperRight); }
  //! Cell bottom right of node \a n
  Cell lowerRight(Node n) const { return _nodeToCell(n, P_LowerRight); }
  //! Cell bottom left of node \a n
  Cell lowerLeft(Node n) const { return _nodeToCell(n, P_LowerLeft); }

  //! Cell top left of node \a n
  ARCCORE_HOST_DEVICE CellLocalId upperLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_UpperLeft); }
  //! Cell top right of node \a n
  ARCCORE_HOST_DEVICE CellLocalId upperRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_UpperRight); }
  //! Cell bottom right of node \a n
  ARCCORE_HOST_DEVICE CellLocalId lowerRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_LowerRight); }
  //! Cell bottom left of node \a n
  ARCCORE_HOST_DEVICE CellLocalId lowerLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_LowerLeft); }

  //! Cell top left of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId upperLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_UpperLeft); }
  //! Cell top right of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId upperRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_UpperRight); }
  //! Cell bottom right of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId lowerRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_LowerRight); }
  //! Cell bottom left of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId lowerLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_LowerLeft); }

  //! In 3D, cell top left of node \a n
  Cell topZUpperLeft(Node n) const { return _nodeToCell(n, P_TopZUpperLeft); }
  //! In 3D, cell top right of node \a n
  Cell topZUpperRight(Node n) const { return _nodeToCell(n, P_TopZUpperRight); }
  //! In 3D, cell bottom right of node \a n
  Cell topZLowerRight(Node n) const { return _nodeToCell(n, P_TopZLowerRight); }
  //! In 3D, cell bottom left of node \a n
  Cell topZLowerLeft(Node n) const { return _nodeToCell(n, P_TopZLowerLeft); }

  //! In 3D, cell top left of node \a n
  ARCCORE_HOST_DEVICE CellLocalId topZUpperLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZUpperLeft); }
  //! In 3D, cell top right of node \a n
  ARCCORE_HOST_DEVICE CellLocalId topZUpperRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZUpperRight); }
  //! In 3D, cell bottom right of node \a n
  ARCCORE_HOST_DEVICE CellLocalId topZLowerRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZLowerRight); }
  //! In 3D, cell bottom left of node \a n
  ARCCORE_HOST_DEVICE CellLocalId topZLowerLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZLowerLeft); }

  //! In 3D, cell top left of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZUpperLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZUpperLeft); }
  //! In 3D, cell top right of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZUpperRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZUpperRight); }
  //! In 3D, cell bottom right of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZLowerRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZLowerRight); }
  //! In 3D, cell bottom left of node \a n for direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZLowerLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZLowerLeft); }

  //! Node top left of cell \a c
  Node upperLeft(Cell c) const { return _cellToNode(c, P_UpperLeft); }
  //! Node top right of cell \a c
  Node upperRight(Cell c) const { return _cellToNode(c, P_UpperRight); }
  //! Node bottom right of cell \a c
  Node lowerRight(Cell c) const { return _cellToNode(c, P_LowerRight); }
  //! Node bottom left of cell \a c
  Node lowerLeft(Cell c) const { return _cellToNode(c, P_LowerLeft); }

  //! Node top left of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId upperLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_UpperLeft); }
  //! Node top right of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId upperRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_UpperRight); }
  //! Node bottom right of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId lowerRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_LowerRight); }
  //! Node bottom left of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId lowerLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_LowerLeft); }

  //! Node top left of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId upperLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_UpperLeft); }
  //! Node top right of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId upperRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_UpperRight); }
  //! Node bottom right of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId lowerRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_LowerRight); }
  //! Node bottom left of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId lowerLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_LowerLeft); }

  //! In 3D, node above top left of cell \a c
  Node topZUpperLeft(Cell c) const { return _cellToNode(c, P_TopZUpperLeft); }
  //! In 3D, node above top right of cell \a c
  Node topZUpperRight(Cell c) const { return _cellToNode(c, P_TopZUpperRight); }
  //! In 3D, node above bottom right of cell \a c
  Node topZLowerRight(Cell c) const { return _cellToNode(c, P_TopZLowerRight); }
  //! In 3D, node above bottom left of cell \a c
  Node topZLowerLeft(Cell c) const { return _cellToNode(c, P_TopZLowerLeft); }

  //! In 3D, node above top left of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZUpperLeft); }
  //! In 3D, node above top right of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZUpperRight); }
  //! In 3D, node above bottom right of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZLowerRight); }
  //! In 3D, node above bottom left of cell \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZLowerLeft); }

  //! In 3D, node above top left of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZUpperLeft); }
  //! In 3D, node above top right of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZUpperRight); }
  //! In 3D, node above bottom right of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZLowerRight); }
  //! In 3D, node above bottom left of cell \a c for direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZLowerLeft); }

 private:

  //! Calculates the connectivity information.
  void _computeInfos(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos(ICartesianMesh* cmesh);
  //! Positions the arrays containing the connectivity information
  void _setStorage(ArrayView<Index> nodes_to_cell, ArrayView<Index> cells_to_node,
                   const Permutation* permutation);

 private:

  ARCCORE_HOST_DEVICE CellLocalId _nodeToCellLocalId(NodeLocalId n, ePosition p) const
  {
    return CellLocalId(m_nodes_to_cell[n].v[p]);
  }
  ARCCORE_HOST_DEVICE NodeLocalId _cellToNodeLocalId(CellLocalId c, ePosition p) const
  {
    return NodeLocalId(m_cells_to_node[c].v[p]);
  }
  ARCCORE_HOST_DEVICE CellLocalId _nodeToCellLocalId(NodeLocalId n, Int32 dir, ePosition p) const
  {
    ARCCORE_CHECK_AT(dir, 3);
    return _nodeToCellLocalId(n, m_permutation->permutation[dir][p]);
  }
  ARCCORE_HOST_DEVICE NodeLocalId _cellToNodeLocalId(CellLocalId c, Int32 dir, ePosition p) const
  {
    ARCCORE_CHECK_AT(dir, 3);
    return _cellToNodeLocalId(c, m_permutation->permutation[dir][p]);
  }
  Cell _nodeToCell(Node n, ePosition p) const { return m_cells[m_nodes_to_cell[n.localId()].v[p]]; }
  Node _cellToNode(Cell c, ePosition p) const { return m_nodes[m_cells_to_node[c.localId()].v[p]]; }

 private:

  // These two methods are for testing
  Index& _index(Node n) { return m_nodes_to_cell[n.localId()]; }
  Index& _index(Cell c) { return m_cells_to_node[c.localId()]; }

 private:

  ArrayView<Index> m_nodes_to_cell;
  ArrayView<Index> m_cells_to_node;
  CellInfoListView m_cells;
  NodeInfoListView m_nodes;
  const Permutation* m_permutation = nullptr;

 private:

  void _computeInfos2D(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos2D(ICartesianMesh* cmesh);
  void _computeInfos3D(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos3D(ICartesianMesh* cmesh);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class for accessing Cartesian connectivities.
 *
 * \sa CartesianConnectivity.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianConnectivityLocalId
: public CartesianConnectivity
{
 private:

  using Index = CartesianConnectivity::Index;

 public:

  CartesianConnectivityLocalId(const CartesianConnectivity& c)
  : CartesianConnectivity(c)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
