// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianConnectivity.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Maillage cartésien.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/cartesianmesh/CartesianConnectivity.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableTypes.h"

#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshNumberingMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_setStorage(ArrayView<Index> nodes_to_cell, ArrayView<Index> cells_to_node,
            const Permutation* permutation)
{
  m_nodes_to_cell = nodes_to_cell;
  m_cells_to_node = cells_to_node;
  m_permutation = permutation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Calcule les infos de connectivité
 *
 * Pour chaque noeud,détermine les 4 mailles autour en fonction des coordonnées
 * des mailles. Fait de même pour les quatres noeuds d'une maille.
 */
void CartesianConnectivity::
_computeInfos(IMesh* mesh, VariableNodeReal3& nodes_coord,
              VariableCellReal3& cells_coord)
{
  m_nodes = NodeInfoListView(mesh->nodeFamily());
  m_cells = CellInfoListView(mesh->cellFamily());

  if (mesh->dimension() == 2 || mesh->dimension() == 1)
    _computeInfos2D(mesh, nodes_coord, cells_coord);
  else if (mesh->dimension() == 3)
    _computeInfos3D(mesh, nodes_coord, cells_coord);
  else
    throw NotSupportedException(A_FUNCINFO, "Unknown mesh dimension");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos(ICartesianMesh* cmesh)
{
  m_nodes = NodeInfoListView(cmesh->mesh()->nodeFamily());
  m_cells = CellInfoListView(cmesh->mesh()->cellFamily());

  if (cmesh->mesh()->dimension() == 2 || cmesh->mesh()->dimension() == 1)
    _computeInfos2D(cmesh);
  else if (cmesh->mesh()->dimension() == 3)
    _computeInfos3D(cmesh);
  else
    throw NotSupportedException(A_FUNCINFO, "Unknown mesh dimension");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos2D(IMesh* mesh, VariableNodeReal3& nodes_coord,
                VariableCellReal3& cells_coord)
{
  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = mesh->nodeFamily();
  IItemFamily* cell_family = mesh->cellFamily();

  ENUMERATE_NODE (inode, node_family->allItems()) {
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    Index& idx = cc._index(node);
    idx.fill(NULL_ITEM_LOCAL_ID);
    Integer nb_cell = node.nbCell();
    for (Integer i = 0; i < nb_cell; ++i) {
      Cell cell = node.cell(i);
      Int32 cell_lid = cell.localId();
      Real3 cell_coord = cells_coord[cell];
      if (cell_coord.y > node_coord.y) {
        if (cell_coord.x > node_coord.x)
          idx.v[P_UpperRight] = cell_lid;
        else
          idx.v[P_UpperLeft] = cell_lid;
      }
      else {
        if (cell_coord.x > node_coord.x)
          idx.v[P_LowerRight] = cell_lid;
        else
          idx.v[P_LowerLeft] = cell_lid;
      }
    }
  }

  ENUMERATE_CELL (icell, cell_family->allItems()) {
    Cell cell = *icell;
    Real3 cell_coord = cells_coord[cell];
    Index& idx = _index(cell);
    idx.fill(NULL_ITEM_LOCAL_ID);
    Integer nb_node = cell.nbNode();
    for (Integer i = 0; i < nb_node; ++i) {
      Node node = cell.node(i);
      Int32 node_lid = node.localId();
      Real3 node_coord = nodes_coord[node];
      if (node_coord.y > cell_coord.y) {
        if (node_coord.x > cell_coord.x)
          idx.v[P_UpperRight] = node_lid;
        else
          idx.v[P_UpperLeft] = node_lid;
      }
      else {
        if (node_coord.x > cell_coord.x)
          idx.v[P_LowerRight] = node_lid;
        else
          idx.v[P_LowerLeft] = node_lid;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos2D(ICartesianMesh* cmesh)
{
  Ref<ICartesianMeshNumberingMngInternal> numbering = cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = cmesh->mesh()->nodeFamily();
  IItemFamily* cell_family = cmesh->mesh()->cellFamily();

  {
    constexpr Integer nb_cell_around_node_max = 4;
    Int64 cells_around[nb_cell_around_node_max];
    ArrayView av_cells_around(nb_cell_around_node_max, cells_around);

    // Le CartesianMeshNumberingMng nous donne toujours les mailles autour du noeud dans le même ordre :
    //
    // |2|3|
    //   .
    // |0|1|
    //
    // y
    // ^
    // |->x
    //
    constexpr Int32 pos_2d[nb_cell_around_node_max] = { P_LowerLeft, P_LowerRight, P_UpperLeft, P_UpperRight };

    ENUMERATE_ (Node, inode, node_family->allItems()) {
      Node node = *inode;
      numbering->cellUniqueIdsAroundNode(av_cells_around, node);

      Index& idx = cc._index(node);
      idx.fill(NULL_ITEM_LOCAL_ID);

      const Integer nb_cell = node.nbCell();
      for (Integer i = 0; i < nb_cell; ++i) {
        Cell cell = node.cell(i);
        Integer pos = 0;
        for (; pos < nb_cell_around_node_max; ++pos) {
          if (cell.uniqueId() == av_cells_around[pos])
            break;
        }
        if (pos == nb_cell_around_node_max)
          continue;

        const Int32 cell_lid = cell.localId();
        idx.v[pos_2d[pos]] = cell_lid;
      }
    }
  }
  {
    constexpr Integer nb_node_in_cell_max = 4;
    Int64 nodes_in_cell[nb_node_in_cell_max];
    ArrayView av_nodes_in_cell(nb_node_in_cell_max, nodes_in_cell);

    // Le CartesianMeshNumberingMng nous donne toujours les noeuds de la maille dans le même ordre :
    //
    // |3|2|
    //   .
    // |0|1|
    //
    // y
    // ^
    // |->x
    //
    constexpr Int32 pos_2d[nb_node_in_cell_max] = { P_LowerLeft, P_LowerRight, P_UpperRight, P_UpperLeft };

    ENUMERATE_ (Cell, icell, cell_family->allItems()) {
      Cell cell = *icell;
      numbering->cellNodeUniqueIds(av_nodes_in_cell, cell);

      Index& idx = _index(cell);
      idx.fill(NULL_ITEM_LOCAL_ID);

      const Integer nb_node = cell.nbNode();
      for (Integer i = 0; i < nb_node; ++i) {
        Node node = cell.node(i);
        Integer pos = 0;
        for (; pos < nb_node_in_cell_max; ++pos) {
          if (node.uniqueId() == av_nodes_in_cell[pos])
            break;
        }
        if (pos == nb_node_in_cell_max)
          continue;

        const Int32 node_lid = node.localId();
        idx.v[pos_2d[pos]] = node_lid;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos3D(IMesh* mesh, VariableNodeReal3& nodes_coord,
                VariableCellReal3& cells_coord)
{
  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = mesh->nodeFamily();
  IItemFamily* cell_family = mesh->cellFamily();

  ENUMERATE_NODE (inode, node_family->allItems()) {
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    Index& idx = cc._index(node);
    idx.fill(NULL_ITEM_LOCAL_ID);
    Integer nb_cell = node.nbCell();
    for (Integer i = 0; i < nb_cell; ++i) {
      Cell cell = node.cell(i);
      Int32 cell_lid = cell.localId();
      Real3 cell_coord = cells_coord[cell];

      if (cell_coord.z > node_coord.z) {
        if (cell_coord.y > node_coord.y) {
          if (cell_coord.x > node_coord.x)
            idx.v[P_TopZUpperRight] = cell_lid;
          else
            idx.v[P_TopZUpperLeft] = cell_lid;
        }
        else {
          if (cell_coord.x > node_coord.x)
            idx.v[P_TopZLowerRight] = cell_lid;
          else
            idx.v[P_TopZLowerLeft] = cell_lid;
        }
      }
      else {
        if (cell_coord.y > node_coord.y) {
          if (cell_coord.x > node_coord.x)
            idx.v[P_UpperRight] = cell_lid;
          else
            idx.v[P_UpperLeft] = cell_lid;
        }
        else {
          if (cell_coord.x > node_coord.x)
            idx.v[P_LowerRight] = cell_lid;
          else
            idx.v[P_LowerLeft] = cell_lid;
        }
      }
    }
  }

  ENUMERATE_CELL (icell, cell_family->allItems()) {
    Cell cell = *icell;
    Real3 cell_coord = cells_coord[cell];
    Index& idx = _index(cell);
    idx.fill(NULL_ITEM_LOCAL_ID);
    Integer nb_node = cell.nbNode();
    for (Integer i = 0; i < nb_node; ++i) {
      Node node = cell.node(i);
      Int32 node_lid = node.localId();
      Real3 node_coord = nodes_coord[node];

      if (node_coord.z > cell_coord.z) {
        if (node_coord.y > cell_coord.y) {
          if (node_coord.x > cell_coord.x)
            idx.v[P_TopZUpperRight] = node_lid;
          else
            idx.v[P_TopZUpperLeft] = node_lid;
        }
        else {
          if (node_coord.x > cell_coord.x)
            idx.v[P_TopZLowerRight] = node_lid;
          else
            idx.v[P_TopZLowerLeft] = node_lid;
        }
      }
      else {
        if (node_coord.y > cell_coord.y) {
          if (node_coord.x > cell_coord.x)
            idx.v[P_UpperRight] = node_lid;
          else
            idx.v[P_UpperLeft] = node_lid;
        }
        else {
          if (node_coord.x > cell_coord.x)
            idx.v[P_LowerRight] = node_lid;
          else
            idx.v[P_LowerLeft] = node_lid;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos3D(ICartesianMesh* cmesh)
{
  Ref<ICartesianMeshNumberingMngInternal> numbering = cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = cmesh->mesh()->nodeFamily();
  IItemFamily* cell_family = cmesh->mesh()->cellFamily();

  {
    constexpr Integer nb_cell_around_node_max = 8;
    Int64 cells_around[nb_cell_around_node_max];
    ArrayView av_cells_around(nb_cell_around_node_max, cells_around);

    // Le CartesianMeshNumberingMng nous donne toujours les mailles autour du noeud dans le même ordre :
    //
    // z = 0 | z = 1
    // |2|3| | |6|7|
    //   .   |   .
    // |0|1| | |4|5|
    //
    // y
    // ^
    // |->x
    //
    constexpr Int32 pos_3d[nb_cell_around_node_max] = { P_LowerLeft, P_LowerRight, P_UpperLeft, P_UpperRight, P_TopZLowerLeft, P_TopZLowerRight, P_TopZUpperLeft, P_TopZUpperRight };

    ENUMERATE_ (Node, inode, node_family->allItems()) {
      Node node = *inode;
      numbering->cellUniqueIdsAroundNode(av_cells_around, node);

      Index& idx = cc._index(node);
      idx.fill(NULL_ITEM_LOCAL_ID);

      const Integer nb_cell = node.nbCell();
      for (Integer i = 0; i < nb_cell; ++i) {
        Cell cell = node.cell(i);
        Integer pos = 0;
        for (; pos < nb_cell_around_node_max; ++pos) {
          if (cell.uniqueId() == av_cells_around[pos])
            break;
        }
        if (pos == nb_cell_around_node_max)
          continue;

        const Int32 cell_lid = cell.localId();
        idx.v[pos_3d[pos]] = cell_lid;
      }
    }
  }
  {
    constexpr Integer nb_node_in_cell_max = 8;
    Int64 nodes_in_cell[nb_node_in_cell_max];
    ArrayView av_nodes_in_cell(nb_node_in_cell_max, nodes_in_cell);

    // Le CartesianMeshNumberingMng nous donne toujours les noeuds de la maille dans le même ordre :
    //
    // z = 0 | z = 1
    // |3|2| | |7|6|
    //   .   |   .
    // |0|1| | |4|5|
    //
    // y
    // ^
    // |->x
    //
    constexpr Int32 pos_3d[nb_node_in_cell_max] = { P_LowerLeft, P_LowerRight, P_UpperRight, P_UpperLeft, P_TopZLowerLeft, P_TopZLowerRight, P_TopZUpperRight, P_TopZUpperLeft };

    ENUMERATE_ (Cell, icell, cell_family->allItems()) {
      Cell cell = *icell;
      numbering->cellNodeUniqueIds(av_nodes_in_cell, cell);

      Index& idx = _index(cell);
      idx.fill(NULL_ITEM_LOCAL_ID);

      const Integer nb_node = cell.nbNode();
      for (Integer i = 0; i < nb_node; ++i) {
        Node node = cell.node(i);
        Integer pos = 0;
        for (; pos < nb_node_in_cell_max; ++pos) {
          if (node.uniqueId() == av_nodes_in_cell[pos])
            break;
        }
        if (pos == nb_node_in_cell_max)
          continue;

        const Int32 node_lid = node.localId();
        idx.v[pos_3d[pos]] = node_lid;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calcule les permutations des 8 ePosition pour chaque direction.
 *
 * La direction de référence est X.
 */
void CartesianConnectivity::Permutation::
compute()
{
  Int32 p[3][8] = {
    { 0, 1, 2, 3, 4, 5, 6, 7 },
    { 3, 0, 1, 2, 7, 4, 5, 6 },
    { 1, 5, 6, 2, 0, 4, 7, 3 }
  };

  for (Int32 i = 0; i < 3; ++i)
    for (Int32 j = 0; j < 8; ++j)
      permutation[i][j] = static_cast<ePosition>(p[i][j]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
