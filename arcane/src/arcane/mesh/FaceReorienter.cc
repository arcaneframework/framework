// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceReorienter.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Vérifie la bonne orientation d'une face et la réoriente le cas échéant.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/mesh/FaceReorienter.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyTopologyModifier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceReorienter::
FaceReorienter(ITraceMng* tm)
: m_trace_mng(tm)
, m_face_family(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceReorienter::
FaceReorienter(IMesh* mesh)
: m_trace_mng(mesh->traceMng())
, m_face_family(nullptr)
{
  m_face_family = mesh->faceFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceReorienter::
checkAndChangeOrientation(ItemInternal* face)
{
  checkAndChangeOrientation(Face(face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceReorienter::
checkAndChangeOrientationAMR(ItemInternal* face)
{
  checkAndChangeOrientationAMR(Face(face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceReorienter::
checkAndChangeOrientation(Face face)
{
  if (!m_face_family)
    m_face_family = face.itemFamily();
  Int32 face_nb_node = face.nbNode();
  IItemFamilyTopologyModifier* face_topology_modifier = m_face_family->_topologyModifier();

  m_nodes_unique_id.resize(face_nb_node);
  m_nodes_local_id.resize(face_nb_node);

  for (Integer i_node = 0; i_node < face_nb_node; ++i_node) {
    m_nodes_unique_id[i_node] = face.node(i_node).uniqueId();
    m_nodes_local_id[i_node] = face.node(i_node).localId();
  }

  m_face_nodes_index.resize(face_nb_node);
  mesh_utils::reorderNodesOfFace2(m_nodes_unique_id, m_face_nodes_index);

  for (Integer i_node = 0; i_node < face_nb_node; ++i_node) {
    ItemLocalId node_lid(m_nodes_local_id[m_face_nodes_index[i_node]]);
    face_topology_modifier->replaceNode(face, i_node, node_lid);
  }

  // On cherche le plus petit uid de la face
  std::pair<Int64, Int64> face_smallest_node_uids = std::make_pair(face.node(0).uniqueId(),
                                                                   face.node(1).uniqueId());
  if (face.node(0).uniqueId() == face.node(1).uniqueId())
    face_smallest_node_uids = std::make_pair(face.node(0).uniqueId(),
                                             face.node(2).uniqueId());
  Cell cell = face.cell(0);
  Int32 cell0_lid = cell.localId();
  Integer local_face_number = -1;
  for (Integer i_face = 0; i_face < cell.nbFace(); ++i_face) {
    if (cell.face(i_face) == face) {
      // On a trouvé la bonne face
      local_face_number = i_face;
      break;
    }
  }

  if (local_face_number == (-1))
    ARCANE_FATAL("Incoherent connectivity: Face {0} not connected to cell {1}",
                 face.uniqueId(), cell.uniqueId());

  const ItemTypeInfo::LocalFace& local_face = cell.typeInfo()->localFace(local_face_number);
  bool cell0_is_back_cell = false;

  if (face_nb_node == 2) {
    cell0_is_back_cell = (cell.node(local_face.node(0)).uniqueId() == face_smallest_node_uids.first);
  }
  else {
    for (Integer i_node = 0; i_node < local_face.nbNode(); ++i_node) {
      if (cell.node(local_face.node(i_node)).uniqueId() == face_smallest_node_uids.first) {
        if (cell.node(local_face.node((i_node + 1) % local_face.nbNode())).uniqueId() == cell.node(local_face.node(i_node)).uniqueId()) {
          if (cell.node(local_face.node((i_node + 2) % local_face.nbNode())).uniqueId() == face_smallest_node_uids.second) {
            cell0_is_back_cell = true;
            break;
          }
          else {
            cell0_is_back_cell = false;
            break;
          }
        }
        else {
          if (cell.node(local_face.node((i_node + 1) % local_face.nbNode())).uniqueId() == face_smallest_node_uids.second) {
            cell0_is_back_cell = true;
            break;
          }
          else {
            cell0_is_back_cell = false;
            break;
          }
        }
      }
    }
  }

  Int32 cell1_lid = (face.nbCell() == 2) ? face.cell(1).localId() : NULL_ITEM_LOCAL_ID;
  // Paire contenant la back_cell et front_cell de la face.
  std::pair<Int32, Int32> face_cells(cell1_lid, cell0_lid);
  if (cell0_is_back_cell) {
    // Si on arrive ici c'est que la maille 0 est la back_cell
    std::swap(face_cells.first, face_cells.second);
  }
  face_topology_modifier->setBackAndFrontCells(face, CellLocalId(face_cells.first), CellLocalId(face_cells.second));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR
void FaceReorienter::
checkAndChangeOrientationAMR(Face face)
{
  if (!m_face_family)
    m_face_family = face.itemFamily();
  IItemFamilyTopologyModifier* face_topology_modifier = m_face_family->_topologyModifier();

  m_nodes_unique_id.resize(face.nbNode());
  m_nodes_local_id.resize(face.nbNode());

  for (Integer i_node = 0; i_node < face.nbNode(); ++i_node) {
    m_nodes_unique_id[i_node] = face.node(i_node).uniqueId();
    m_nodes_local_id[i_node] = face.node(i_node).localId();
  }

  m_face_nodes_index.resize(face.nbNode());
  mesh_utils::reorderNodesOfFace2(m_nodes_unique_id, m_face_nodes_index);

  for (Integer i_node = 0; i_node < face.nbNode(); ++i_node) {
    ItemLocalId node_lid(m_nodes_local_id[m_face_nodes_index[i_node]]);
    face_topology_modifier->replaceNode(face, i_node, node_lid);
  }

  // On cherche le plus petit uid de la face
  std::pair<Int64, Int64> face_smallest_node_uids = std::make_pair(face.node(0).uniqueId(),
                                                                   face.node(1).uniqueId());

  if (face.node(0).uniqueId() == face.node(1).uniqueId())
    face_smallest_node_uids = std::make_pair(face.node(0).uniqueId(),
                                             face.node(2).uniqueId());

  //ItemInternal* cell = face->cell(0);
  bool cell_0 = false;
  bool cell_1 = false;
  Cell cell;

  if (face.nbCell() == 2) {
    if (face.cell(0).level() >= face.cell(1).level()) {
      cell = face.cell(0);
      cell_0 = true;
    }
    else {
      cell = face.cell(1);
      cell_1 = true;
    }
  }
  else {
    cell = face.cell(0);
    cell_0 = true;
    if (cell.null()) {
      ARCANE_FATAL("Face without cells cannot be possible -- Face uid: {0}", face.uniqueId());
    }
  }
  Integer local_face_number = -1;
  for (Integer i_face = 0; i_face < cell.nbFace(); ++i_face) {
    if (cell.face(i_face) == face) {
      // On a trouvé la bonne face
      local_face_number = i_face;
      break;
    }
  }

  if (local_face_number == (-1)) {
    ARCANE_FATAL("Incoherent connectivity: Face {0} not connected to cell {1}",
                 face.uniqueId(), cell.uniqueId());
  }

  const ItemTypeInfo::LocalFace& local_face = cell.typeInfo()->localFace(local_face_number);

  bool cell_is_back_cell = false;

  if (face.nbNode() == 2) {
    cell_is_back_cell = (cell.node(local_face.node(0)).uniqueId() == face_smallest_node_uids.first);
  }
  else {
    for (Integer i_node = 0; i_node < local_face.nbNode(); ++i_node) {
      if (cell.node(local_face.node(i_node)).uniqueId() == face_smallest_node_uids.first) {
        if (cell.node(local_face.node((i_node + 1) % local_face.nbNode())).uniqueId() == cell.node(local_face.node(i_node)).uniqueId()) {
          if (cell.node(local_face.node((i_node + 2) % local_face.nbNode())).uniqueId() == face_smallest_node_uids.second) {
            cell_is_back_cell = true;
            break;
          }
          else {
            cell_is_back_cell = false;
            break;
          }
        }
        else {
          if (cell.node(local_face.node((i_node + 1) % local_face.nbNode())).uniqueId() == face_smallest_node_uids.second) {
            cell_is_back_cell = true;
            break;
          }
          else {
            cell_is_back_cell = false;
            break;
          }
        }
      }
    }
  }

  // Paire contenant la back_cell et front_cell de la face.
  std::pair<Int32, Int32> face_cells(NULL_ITEM_LOCAL_ID, NULL_ITEM_LOCAL_ID);
  bool face_has_two_cell = (face.nbCell() == 2);

  if (cell_0) {
    Int32 cell1_lid = (face_has_two_cell) ? face.cell(1).localId() : NULL_ITEM_LOCAL_ID;
    if (cell_is_back_cell) {
      // Si on arrive ici, c'est que la maille 0 est la back_cell
      // La front cell est toujours cell1_lid (qui peut être nulle).
      face_cells = { face.cell(0).localId(), cell1_lid };
    }
    else {
      // Si on arrive ici, c'est que la maille 0 est la front_cell
      // La back cell est toujours cell1_lid (qui peut être nulle)
      face_cells.first = cell1_lid;
      face_cells.second = (face_has_two_cell) ? cell.localId() : face.cell(0).localId();
    }
  }
  else if (cell_1) {
    if (cell_is_back_cell) {
      // Si on arrive ici, c'est que la maille 0 est la front_cell
      // On met à jour les infos d'orientation
      face_cells.second = face.cell(0).localId();
      // GG Attention, si ici, cela signifie qu'il faut échanger la front cell
      // et la back cell car la back cell doit toujours être la première
      face_cells.first = (face_has_two_cell) ? cell.localId() : NULL_ITEM_LOCAL_ID;
    }
    else {
      // Si on arrive ici, c'est que la maille 0 est la back_cell
      // On met à jour les infos d'orientation
      face_cells.first = face.cell(0).localId();
      face_cells.second = (face_has_two_cell) ? face.cell(1).localId() : NULL_ITEM_LOCAL_ID;
    }
  }
  face_topology_modifier->setBackAndFrontCells(face, CellLocalId(face_cells.first), CellLocalId(face_cells.second));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
