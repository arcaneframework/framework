// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshConnectivity.h                              (C) 2000-2025 */
/*                                                                           */
/* Informations de connectivité d'un maillage non structuré.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_UNSTRUCTUREDMESHCONNECTIVITY_H
#define ARCANE_CORE_UNSTRUCTUREDMESHCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IndexedItemConnectivityView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les connectivités standards d'un maillage non structuré.
 *
 * Il faut appeler setMesh() avant d'utiliser les méthodes de cette classe.
 * La méthode setMesh() doit être appelée si la cardinalité du maillage évolue.
 */
class ARCANE_CORE_EXPORT UnstructuredMeshConnectivityView
{
 public:

  UnstructuredMeshConnectivityView() = default;
  explicit UnstructuredMeshConnectivityView(IMesh* m){ setMesh(m); }

 public:

 void setMesh(IMesh* m);

 public:

  IndexedCellNodeConnectivityView cellNode() const
  { _checkValid(); return m_cell_node_connectivity_view; }
  IndexedCellEdgeConnectivityView cellEdge() const
  { _checkValid(); return m_cell_edge_connectivity_view; }
  IndexedCellFaceConnectivityView cellFace() const
  { _checkValid(); return m_cell_face_connectivity_view; }

  IndexedFaceNodeConnectivityView faceNode() const
  { _checkValid(); return m_face_node_connectivity_view; }
  IndexedFaceEdgeConnectivityView faceEdge() const
  { _checkValid(); return m_face_edge_connectivity_view; }
  IndexedFaceCellConnectivityView faceCell() const
  { _checkValid(); return m_face_cell_connectivity_view; }

  IndexedNodeEdgeConnectivityView nodeEdge() const
  { _checkValid(); return m_node_edge_connectivity_view; }
  IndexedNodeFaceConnectivityView nodeFace() const
  { _checkValid(); return m_node_face_connectivity_view; }
  IndexedNodeCellConnectivityView nodeCell() const
  { _checkValid(); return m_node_cell_connectivity_view; }

  IndexedEdgeNodeConnectivityView edgeNode() const
  { _checkValid(); return m_edge_node_connectivity_view; }
  IndexedEdgeFaceConnectivityView edgeFace() const
  { _checkValid(); return m_edge_face_connectivity_view; }
  IndexedEdgeCellConnectivityView edgeCell() const
  { _checkValid(); return m_edge_cell_connectivity_view; }

 private:

  IndexedCellNodeConnectivityView m_cell_node_connectivity_view;
  IndexedCellEdgeConnectivityView m_cell_edge_connectivity_view;
  IndexedCellFaceConnectivityView m_cell_face_connectivity_view;

  IndexedFaceNodeConnectivityView m_face_node_connectivity_view;
  IndexedFaceEdgeConnectivityView m_face_edge_connectivity_view;
  IndexedFaceCellConnectivityView m_face_cell_connectivity_view;

  IndexedNodeEdgeConnectivityView m_node_edge_connectivity_view;
  IndexedNodeFaceConnectivityView m_node_face_connectivity_view;
  IndexedNodeCellConnectivityView m_node_cell_connectivity_view;

  IndexedEdgeNodeConnectivityView m_edge_node_connectivity_view;
  IndexedEdgeFaceConnectivityView m_edge_face_connectivity_view;
  IndexedEdgeCellConnectivityView m_edge_cell_connectivity_view;

  IMesh* m_mesh = nullptr;

 private:

  void _checkValid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
