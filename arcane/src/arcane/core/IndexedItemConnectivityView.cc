// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedItemConnectivityView.cc                              (C) 2000-2026 */
/*                                                                           */
/* Vues sur les connectivités utilisant des index.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IndexedItemConnectivityView.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Fonction utilisée pour tester la compilation avec accès aux connectivités

extern "C++" void
_internalItemTestCompile()
{
  IndexedCellNodeConnectivityView cell_node;
  IndexedCellEdgeConnectivityView cell_edge;
  IndexedCellFaceConnectivityView cell_face;
  IndexedCellCellConnectivityView cell_cell;
  IndexedCellDoFConnectivityView cell_dof;

  IndexedFaceNodeConnectivityView face_node;
  IndexedFaceEdgeConnectivityView face_edge;
  IndexedFaceFaceConnectivityView face_face;
  IndexedFaceCellConnectivityView face_cell;
  IndexedFaceDoFConnectivityView face_dof;

  IndexedEdgeNodeConnectivityView edge_node;
  IndexedEdgeEdgeConnectivityView edge_edge;
  IndexedEdgeFaceConnectivityView edge_face;
  IndexedEdgeCellConnectivityView edge_cell;
  IndexedEdgeDoFConnectivityView edge_dof;

  IndexedNodeNodeConnectivityView node_node;
  IndexedNodeEdgeConnectivityView node_edge;
  IndexedNodeFaceConnectivityView node_face;
  IndexedNodeCellConnectivityView node_cell;
  IndexedNodeDoFConnectivityView node_dof;

  IndexedDoFNodeConnectivityView dof_node;
  IndexedDoFEdgeConnectivityView dof_edge;
  IndexedDoFFaceConnectivityView dof_face;
  IndexedDoFCellConnectivityView dof_cell;
  IndexedDoFDoFConnectivityView dof_dof;

  ItemGroup items;
  Int64 total = 0;

  ENUMERATE_ (Node, inode, items) {
    Node xnode = *inode;
    for (NodeLocalId node : node_node.nodes(xnode)) {
      total += node.localId();
    }
    for (EdgeLocalId edge : node_edge.edges(xnode)) {
      total += edge.localId();
    }
    for (FaceLocalId face : node_face.faces(xnode)) {
      total += face.localId();
    }
    for (CellLocalId cell : node_cell.cells(xnode)) {
      total += cell.localId();
    }
    for (DoFLocalId dof : node_dof.dofs(xnode)) {
      total += dof.localId();
    }
  }

  ENUMERATE_ (Cell, icell, items) {
    Cell xcell = *icell;
    for (NodeLocalId node : cell_node.nodes(xcell)) {
      total += node.localId();
    }
    for (EdgeLocalId edge : cell_edge.edges(xcell)) {
      total += edge.localId();
    }
    for (FaceLocalId face : cell_face.faces(xcell)) {
      total += face.localId();
    }
    for (CellLocalId cell : cell_cell.cells(xcell)) {
      total += cell.localId();
    }
    for (DoFLocalId dof : cell_dof.dofs(xcell)) {
      total += dof.localId();
    }
  }

  ENUMERATE_ (Edge, iedge, items) {
    Edge xedge = *iedge;
    for (NodeLocalId node : edge_node.nodes(xedge)) {
      total += node.localId();
    }
    for (EdgeLocalId edge : edge_edge.edges(xedge)) {
      total += edge.localId();
    }
    for (FaceLocalId face : edge_face.faces(xedge)) {
      total += face.localId();
    }
    for (CellLocalId cell : edge_cell.cells(xedge)) {
      total += cell.localId();
    }
    for (DoFLocalId dof : edge_dof.dofs(xedge)) {
      total += dof.localId();
    }
  }

  ENUMERATE_ (Face, iface, items) {
    Face xface = *iface;
    for (NodeLocalId node : face_node.nodes(xface)) {
      total += node.localId();
    }
    for (EdgeLocalId edge : face_edge.edges(xface)) {
      total += edge.localId();
    }
    for (FaceLocalId face : face_face.faces(xface)) {
      total += face.localId();
    }
    for (CellLocalId cell : face_cell.cells(xface)) {
      total += cell.localId();
    }
    for (DoFLocalId dof : face_dof.dofs(xface)) {
      total += dof.localId();
    }
  }

  ENUMERATE_ (DoF, idof, items) {
    DoF xdof = *idof;
    for (NodeLocalId node : dof_node.nodes(xdof)) {
      total += node.localId();
    }
    for (EdgeLocalId edge : dof_edge.edges(xdof)) {
      total += edge.localId();
    }
    for (FaceLocalId face : dof_face.faces(xdof)) {
      total += face.localId();
    }
    for (CellLocalId cell : dof_cell.cells(xdof)) {
      total += cell.localId();
    }
    for (DoFLocalId dof : dof_dof.dofs(xdof)) {
      total += dof.localId();
    }
  }

  std::cout << "TOTAL=" << total << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedParticleCellConnectivityView::
IndexedParticleCellConnectivityView(IParticleFamily* pf)
: IndexedParticleCellConnectivityView(pf->itemFamily())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedParticleCellConnectivityView::
IndexedParticleCellConnectivityView(IItemFamily* family)
{
  ItemInternalConnectivityList* clist = family->_internalApi()->unstructuredItemInternalConnectivityList();
  m_container_view = clist->containerView(IK_Cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
