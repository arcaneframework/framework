// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshArea.h                                                  (C) 2000-2016 */
/*                                                                           */
/* Accès aux informations d'un maillage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/MeshArea.h"
#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshArea::
MeshArea(IMesh* mesh)
: m_mesh(mesh)
, m_sub_domain(mesh->subDomain())
, m_trace_mng(m_sub_domain->traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshArea::
~MeshArea()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* MeshArea::
subDomain()
{
  return m_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* MeshArea::
traceMng()
{
  return m_trace_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshArea::
mesh()
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshArea::
setArea(const NodeGroup& nodes,const CellGroup& cells)
{
  m_all_nodes = nodes;
  m_all_cells = cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshArea::nbNode() { return m_all_nodes.size(); }
Integer MeshArea::nbEdge() { return m_all_edges.size(); }
Integer MeshArea::nbFace() { return m_all_faces.size(); }
Integer MeshArea::nbCell() { return m_all_cells.size(); }
Integer MeshArea::nbItem(eItemKind ik)
{
  switch(ik){
  case IK_Node: return nbNode();
  case IK_Edge: return nbEdge();
  case IK_Face: return nbFace();
  case IK_Cell: return nbCell();
  case IK_Particle:
  case IK_DualNode:
  case IK_Link:
  case IK_DoF:
  case IK_Unknown:
    break;
  }
  throw FatalErrorException(A_FUNCINFO,"invalid argument");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup MeshArea::allNodes() { return m_all_nodes; }
EdgeGroup MeshArea::allEdges() { return m_all_edges; }
FaceGroup MeshArea::allFaces() { return m_all_faces; }
CellGroup MeshArea::allCells() { return m_all_cells; }
ItemGroup MeshArea::allItems(eItemKind ik)
{
  switch(ik){
  case IK_Node: return allNodes();
  case IK_Edge: return allEdges();
  case IK_Face: return allFaces();
  case IK_Cell: return allCells();
  case IK_Particle:
  case IK_Link:
  case IK_DualNode:
  case IK_DoF:
  case IK_Unknown:
    break;
  }
  throw FatalErrorException(A_FUNCINFO,"invalid argument");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup MeshArea::ownNodes() { return m_all_nodes.own(); }
EdgeGroup MeshArea::ownEdges() { return m_all_edges.own(); }
FaceGroup MeshArea::ownFaces() { return m_all_faces.own(); }
CellGroup MeshArea::ownCells() { return m_all_cells.own(); }
ItemGroup MeshArea::ownItems(eItemKind ik)
{
  switch(ik){
  case IK_Node: return ownNodes();
  case IK_Edge: return ownEdges();
  case IK_Face: return ownFaces();
  case IK_Cell: return ownCells();
  case IK_Particle:
  case IK_DualNode:
  case IK_Link:
  case IK_DoF:
  case IK_Unknown:
    break;
  }
  throw FatalErrorException(A_FUNCINFO,"invalid argument");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
