// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianConnectivity.cc                                    (C) 2000-2021 */
/*                                                                           */
/* Maillage cartésien.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/cartesianmesh/CartesianConnectivity.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
setStorage(ArrayView<Index> nodes_to_cell,ArrayView<Index> cells_to_node)
{
  m_nodes_to_cell = nodes_to_cell;
  m_cells_to_node = cells_to_node;
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
computeInfos(IMesh* mesh,VariableNodeReal3& nodes_coord,
             VariableCellReal3& cells_coord)
{
  if (mesh->dimension()==2 || mesh->dimension()==1)
    _computeInfos2D(mesh,nodes_coord,cells_coord);
  else if (mesh->dimension()==3)
    _computeInfos3D(mesh,nodes_coord,cells_coord);
  else
    throw NotSupportedException(A_FUNCINFO,"Unknown mesh dimension");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos2D(IMesh* mesh,VariableNodeReal3& nodes_coord,
                VariableCellReal3& cells_coord)
{
  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = mesh->nodeFamily();  
  IItemFamily* cell_family = mesh->cellFamily();

  ItemInternal* null_ii = ItemInternal::nullItem();

  ENUMERATE_NODE(inode,node_family->allItems()){
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    Index& idx = cc._index(node);
    idx.fill(null_ii);
    Integer nb_cell = node.nbCell();
    for( Integer i=0; i<nb_cell; ++i ){
      Cell cell = node.cell(i);
      ItemInternal* cell_ii = cell.internal();
      Real3 cell_coord = cells_coord[cell];
      if (cell_coord.y > node_coord.y){
        if (cell_coord.x > node_coord.x)
          idx.v[P_UpperRight] = cell_ii;
        else
          idx.v[P_UpperLeft] = cell_ii;
      }
      else{
        if (cell_coord.x > node_coord.x)
          idx.v[P_LowerRight] = cell_ii;
        else
          idx.v[P_LowerLeft] = cell_ii;
      }
    }
  }

  ENUMERATE_CELL(icell,cell_family->allItems()){
    Cell cell = *icell;
    Real3 cell_coord = cells_coord[cell];
    Index& idx = _index(cell);
    idx.fill(null_ii);
    Integer nb_node = cell.nbNode();
    for( Integer i=0; i<nb_node; ++i ){
      Node node = cell.node(i);
      ItemInternal* node_ii = node.internal();
      Real3 node_coord = nodes_coord[node];
      if (node_coord.y > cell_coord.y){
        if (node_coord.x > cell_coord.x)
          idx.v[P_UpperRight] = node_ii;
        else
          idx.v[P_UpperLeft] = node_ii;
      }
      else{
        if (node_coord.x > cell_coord.x)
          idx.v[P_LowerRight] = node_ii;
        else
          idx.v[P_LowerLeft] = node_ii;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianConnectivity::
_computeInfos3D(IMesh* mesh,VariableNodeReal3& nodes_coord,
                VariableCellReal3& cells_coord)
{
  CartesianConnectivity& cc = *this;
  IItemFamily* node_family = mesh->nodeFamily();  
  IItemFamily* cell_family = mesh->cellFamily();

  ItemInternal* null_ii = ItemInternal::nullItem();

  ENUMERATE_NODE(inode,node_family->allItems()){
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    Index& idx = cc._index(node);
    idx.fill(null_ii);
    Integer nb_cell = node.nbCell();
    for( Integer i=0; i<nb_cell; ++i ){
      Cell cell = node.cell(i);
      ItemInternal* cell_ii = cell.internal();
      Real3 cell_coord = cells_coord[cell];

      if (cell_coord.z > node_coord.z){
        if (cell_coord.y > node_coord.y){
          if (cell_coord.x > node_coord.x)
            idx.v[P_TopZUpperRight] = cell_ii;
          else
            idx.v[P_TopZUpperLeft] = cell_ii;
        }
        else{
          if (cell_coord.x > node_coord.x)
            idx.v[P_TopZLowerRight] = cell_ii;
          else
            idx.v[P_TopZLowerLeft] = cell_ii;
        }
      }
      else{
        if (cell_coord.y > node_coord.y){
          if (cell_coord.x > node_coord.x)
            idx.v[P_UpperRight] = cell_ii;
          else
            idx.v[P_UpperLeft] = cell_ii;
        }
        else{
          if (cell_coord.x > node_coord.x)
            idx.v[P_LowerRight] = cell_ii;
          else
            idx.v[P_LowerLeft] = cell_ii;
        }
      }
    }
  }

  ENUMERATE_CELL(icell,cell_family->allItems()){
    Cell cell = *icell;
    Real3 cell_coord = cells_coord[cell];
    Index& idx = _index(cell);
    idx.fill(null_ii);
    Integer nb_node = cell.nbNode();
    for( Integer i=0; i<nb_node; ++i ){
      Node node = cell.node(i);
      ItemInternal* node_ii = node.internal();
      Real3 node_coord = nodes_coord[node];

      if (node_coord.z > cell_coord.z){
        if (node_coord.y > cell_coord.y){
          if (node_coord.x > cell_coord.x)
            idx.v[P_TopZUpperRight] = node_ii;
          else
            idx.v[P_TopZUpperLeft] = node_ii;
        }
        else{
          if (node_coord.x > cell_coord.x)
            idx.v[P_TopZLowerRight] = node_ii;
          else
            idx.v[P_TopZLowerLeft] = node_ii;
        }
      }
      else{
        if (node_coord.y > cell_coord.y){
          if (node_coord.x > cell_coord.x)
            idx.v[P_UpperRight] = node_ii;
          else
            idx.v[P_UpperLeft] = node_ii;
        }
        else{
          if (node_coord.x > cell_coord.x)
            idx.v[P_LowerRight] = node_ii;
          else
            idx.v[P_LowerLeft] = node_ii;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
