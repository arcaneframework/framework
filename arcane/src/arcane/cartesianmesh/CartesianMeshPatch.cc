// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshPatch.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"

#include "arcane/IMesh.h"
#include "arcane/ItemPrinter.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianConnectivity.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshPatch::
CartesianMeshPatch(ICartesianMesh* cmesh,Integer patch_index)
: TraceAccessor(cmesh->traceMng())
, m_mesh(cmesh)
, m_amr_patch_index(patch_index)
{
  Integer nb_dir = cmesh->mesh()->dimension();
  for( Integer i=0; i<nb_dir; ++i ){
    eMeshDirection dir = static_cast<eMeshDirection>(i);
    m_cell_directions[i]._internalInit(cmesh,dir,patch_index);
    m_face_directions[i]._internalInit(cmesh,dir,patch_index);
    m_node_directions[i]._internalInit(cmesh,dir,patch_index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshPatch::
~CartesianMeshPatch()
{
  for( Integer i=0; i<3; ++i ){
    m_cell_directions[i]._internalDestroy();
    m_face_directions[i]._internalDestroy();
    m_node_directions[i]._internalDestroy();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianMeshPatch::
cells()
{
  // Le groupe de mailles du patch est le même dans toutes les directions.
  return cellDirection(MD_DirX).allCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Calcule les infos sur les noeuds avant/après et gauche/droite d'une maille
 * pour chaque direction.
 */
void CartesianMeshPatch::
_computeNodeCellInformations2D(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord)
{
  using Int8 = std::int8_t;
  Int8 nodes_indirection_i[CellDirectionMng::MAX_NB_NODE];
  ArrayView<Int8> nodes_indirection(CellDirectionMng::MAX_NB_NODE,nodes_indirection_i);
  Int32 nb_node = cell0.nbNode();
  if (nb_node!=4)
    ARCANE_FATAL("Number of nodes should be '4' (v={0})",nb_node);
  Int8 i8_nb_node = 4;
  Real3 cell_coord = cell0_coord;
  bool is_2d = m_mesh->mesh()->dimension()==2;
  if (!is_2d)
    ARCANE_FATAL("Invalid call. This mesh is not a 2D mesh");

  // Direction X
  nodes_indirection.fill(-1);
  for( Int8 i=0; i<i8_nb_node; ++i ){
    Node node = cell0.node(i);
    Real3 node_coord = nodes_coord[node];
    if (node_coord.x>cell_coord.x){
      if (node_coord.y>cell_coord.y)
        nodes_indirection[CNP_NextLeft] = i;
      else
        nodes_indirection[CNP_NextRight] = i;
    }
    else{
      if (node_coord.y>cell_coord.y)
        nodes_indirection[CNP_PreviousLeft] = i;
      else
        nodes_indirection[CNP_PreviousRight] = i;
    }
  }
  cellDirection(MD_DirX).setNodesIndirection(nodes_indirection);

  // Direction Y
  nodes_indirection.fill(-1);
  for( Int8 i=0; i<i8_nb_node; ++i ){
    Node node = cell0.node(i);
    Real3 node_coord = nodes_coord[node];
    if (node_coord.y>cell_coord.y){
      if (node_coord.x>cell_coord.x)
        nodes_indirection[CNP_NextRight] = i;
      else
        nodes_indirection[CNP_NextLeft] = i;
    }
    else{
      if (node_coord.x>cell_coord.x)
        nodes_indirection[CNP_PreviousRight] = i;
      else
        nodes_indirection[CNP_PreviousLeft] = i;
    }
  }
  cellDirection(MD_DirY).setNodesIndirection(nodes_indirection);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Calcule les infos sur les noeuds avant/après et gauche/droite d'une maille
 * pour chaque direction.
 */
void CartesianMeshPatch::
_computeNodeCellInformations3D(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord)
{
  using Int8 = std::int8_t;
  Int8 nodes_indirection_i[CellDirectionMng::MAX_NB_NODE];
  ArrayView<Int8> nodes_indirection(CellDirectionMng::MAX_NB_NODE,nodes_indirection_i);
  Integer nb_node = cell0.nbNode();
  if (nb_node!=8)
    ARCANE_FATAL("Number of nodes should be '8' (v={0})",nb_node);
  Int8 i8_nb_node = 8;
  Real3 cell_coord = cell0_coord;
  bool is_3d = m_mesh->mesh()->dimension()==3;
  if (!is_3d)
    ARCANE_FATAL("Invalid call. This mesh is not a 3D mesh");

  // Direction X
  nodes_indirection.fill(-1);
  for( Int8 i=0; i<i8_nb_node; ++i ){
    Node node = cell0.node(i);
    Real3 node_coord = nodes_coord[node];
    if (node_coord.z>cell_coord.z){
      if (node_coord.x>cell_coord.x){
        if (node_coord.y>cell_coord.y)
          nodes_indirection[CNP_TopNextLeft] = i;
        else
          nodes_indirection[CNP_TopNextRight] = i;
      }
      else{
        if (node_coord.y>cell_coord.y)
          nodes_indirection[CNP_TopPreviousLeft] = i;
        else
          nodes_indirection[CNP_TopPreviousRight] = i;
      }
    }
    else{
      if (node_coord.x>cell_coord.x){
        if (node_coord.y>cell_coord.y)
          nodes_indirection[CNP_NextLeft] = i;
        else
          nodes_indirection[CNP_NextRight] = i;
      }
      else{
        if (node_coord.y>cell_coord.y)
          nodes_indirection[CNP_PreviousLeft] = i;
        else
          nodes_indirection[CNP_PreviousRight] = i;
      }
    }
  }
  cellDirection(MD_DirX).setNodesIndirection(nodes_indirection);

  // Direction Y
  nodes_indirection.fill(-1);
  for( Int8 i=0; i<i8_nb_node; ++i ){
    Node node = cell0.node(i);
    Real3 node_coord = nodes_coord[node];
    if (node_coord.z>cell_coord.z){
      if (node_coord.y>cell_coord.y){
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_TopNextRight] = i;
        else
          nodes_indirection[CNP_TopNextLeft] = i;
      }
      else{
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_TopPreviousRight] = i;
        else
          nodes_indirection[CNP_TopPreviousLeft] = i;
      }
    }
    else{
      if (node_coord.y>cell_coord.y){
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_NextRight] = i;
        else
          nodes_indirection[CNP_NextLeft] = i;
      }
      else{
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_PreviousRight] = i;
        else
          nodes_indirection[CNP_PreviousLeft] = i;
      }
    }
  }
  cellDirection(MD_DirY).setNodesIndirection(nodes_indirection);

  nodes_indirection.fill(-1);
  for( Int8 i=0; i<i8_nb_node; ++i ){
    Node node = cell0.node(i);
    Real3 node_coord = nodes_coord[node];
    if (node_coord.y>cell_coord.y){
      if (node_coord.z>cell_coord.z){
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_TopNextRight] = i;
        else
          nodes_indirection[CNP_TopNextLeft] = i;
      }
      else{
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_TopPreviousRight] = i;
        else
          nodes_indirection[CNP_TopPreviousLeft] = i;
      }
    }
    else{
      if (node_coord.z>cell_coord.z){
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_NextRight] = i;
        else
          nodes_indirection[CNP_NextLeft] = i;
      }
      else{
        if (node_coord.x>cell_coord.x)
          nodes_indirection[CNP_PreviousRight] = i;
        else
          nodes_indirection[CNP_PreviousLeft] = i;
      }
    }
  }
  cellDirection(MD_DirZ).setNodesIndirection(nodes_indirection);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshPatch::
_internalComputeNodeCellInformations(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord)
{
  int dim = m_mesh->mesh()->dimension();
  if (dim==3)
    _computeNodeCellInformations3D(cell0,cell0_coord,nodes_coord);
  else if (dim==2)
    _computeNodeCellInformations2D(cell0,cell0_coord,nodes_coord);
  else
    ARCANE_THROW(NotImplementedException,"this method is implemented only for 2D and 3D mesh (dim={0})",dim);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshPatch::
checkValid() const
{
  // Vérifie que toutes les mailles avant/après appartiennent au groupe de
  // mailles de la direction
  Integer nb_dir = m_mesh->mesh()->dimension();
  for( Integer i=0; i<nb_dir; ++i ){
    CellDirectionMng dm = m_cell_directions[i];
    FaceDirectionMng face_dm = m_face_directions[i];
    std::set<Int32>  cells_ids;
    CellGroup dm_cells = dm.allCells();
    info(4) << "PATCH i=" << m_amr_patch_index << " nb_cell=" << dm_cells.size();
    ENUMERATE_CELL(icell,dm_cells){
      cells_ids.insert(icell.itemLocalId());
    }
    Int64 nb_null_face_cell = 0;
    ENUMERATE_CELL(icell,dm_cells){
      Cell cell = *icell;
      DirCell cc(dm.cell(cell));
      Cell next_cell = cc.next();
      if (!next_cell.null()){
        if (cells_ids.find(next_cell.localId())==cells_ids.end())
          ARCANE_FATAL("Bad next cell dir={0} cell={1} next={2}",i,ItemPrinter(cell),ItemPrinter(next_cell));
      }
      Cell previous_cell = cc.previous();
      if (!previous_cell.null()){
        if (cells_ids.find(previous_cell.localId())==cells_ids.end())
          ARCANE_FATAL("Bad previous cell dir={0} cell={1} previous={2}",i,ItemPrinter(cell),ItemPrinter(previous_cell));
      }
      // Regarde si les infos des faces sont valides

      DirCellFace cell_face(dm.cellFace(cell));
      Face prev_face = cell_face.previous();
      Face next_face = cell_face.next();
      DirFace dir_face_prev(face_dm[prev_face]);
      Cell face_cell_prev = dir_face_prev.previousCell();
      if (face_cell_prev.null())
        ++nb_null_face_cell;
      DirFace dir_face_next(face_dm[next_face]);
      Cell face_cell_next = dir_face_next.nextCell();
      if (face_cell_next.null())
        ++nb_null_face_cell;
    }
    info(4) << "PATCH i=" << m_amr_patch_index << " nb_null_face_cell=" << nb_null_face_cell;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
