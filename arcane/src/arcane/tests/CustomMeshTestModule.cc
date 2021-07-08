// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CustomMeshTestModule.cc                          C) 2000-2021             */
/*                                                                           */
/* Test Module for custom mesh                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ITimeLoopMng.h"
#include "arcane/IMesh.h"
#include "arcane/MeshHandle.h"
#include "arcane/IMeshMng.h"
#include "arcane/mesh/PolyhedralMesh.h"

#include "CustomMeshTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::CustomMesh {
using namespace Arcane;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CustomMeshTestModule : public ArcaneCustomMeshTestObject {
 public:
  CustomMeshTestModule (const ModuleBuildInfo& sbi) : ArcaneCustomMeshTestObject(sbi){}

 public:
  void init() {
    info() << "-- INIT CUSTOM MESH MODULE";
    auto mesh_handle = subDomain()->meshMng()->findMeshHandle(mesh::PolyhedralMesh::handleName());
    if (mesh_handle.hasMesh()) {
      _testEnumerationAndConnectivities(mesh_handle.mesh());
      _testVariableAndGroups(mesh_handle.mesh());
    }
    else info() << "No Mesh";

    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }

 private:
  void _testEnumerationAndConnectivities(IMesh* mesh);
  void _testVariableAndGroups(IMesh* mesh);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CustomMeshTestModule::
_testEnumerationAndConnectivities(IMesh* mesh)
{
  info() << "- Polyhedral mesh test -";
  info() << "- Mesh dimension " << mesh->dimension();
  info() << "- Mesh nb cells  " << mesh->nbItem(IK_Cell) << " or " << mesh->nbCell();
  info() << "- Mesh nb faces  " << mesh->nbItem(IK_Face) << " or " << mesh->nbFace();
  info() << "- Mesh nb edges  " << mesh->nbItem(IK_Edge) << " or " << mesh->nbEdge();
  info() << "- Mesh nb nodes  " << mesh->nbItem(IK_Node) << " or " << mesh->nbNode();
  info() << "Cell family " << mesh->cellFamily()->name();
  info() << "Node family " << mesh->nodeFamily()->name();
  auto all_cells = mesh->allCells();
  // ALL CELLS
  ENUMERATE_CELL (icell, all_cells) {
    info() << "cell with index " << icell.index();
    info() << "cell with lid " << icell.localId();
    info() << "cell with uid " << icell->uniqueId().asInt64();
    info() << "cell number of nodes " << icell->nodes().size();
    info() << "cell number of faces " << icell->faces().size();
    info() << "cell number of edges " << icell->edges().size();
    ENUMERATE_NODE (inode, icell->nodes()) {
      info() << "cell node " << inode.index() << " lid " << inode.localId() << " uid " << inode->uniqueId().asInt64();
    }
    ENUMERATE_FACE (iface, icell->faces()) {
      info() << "cell face " << iface.index() << " lid " << iface.localId() << " uid " << iface->uniqueId().asInt64();
    }
    ENUMERATE_EDGE (iedge, icell->edges()) {
      info() << "cell edge " << iedge.index() << " lid " << iedge.localId() << " uid " << iedge->uniqueId().asInt64();
    }
  }
  // ALL FACES
  ENUMERATE_FACE (iface, mesh->allFaces()) {
    info() << "face with index " << iface.index();
    info() << "face with lid " << iface.localId();
    info() << "face with uid " << iface->uniqueId().asInt64();
    info() << "face number of nodes " << iface->nodes().size();
    info() << "face number of cells " << iface->cells().size();
    info() << "face number of edges " << iface->edges().size();
    ENUMERATE_NODE (inode, iface->nodes()) {
      info() << "face node " << inode.index() << " lid " << inode.localId() << " uid " << inode->uniqueId().asInt64();
    }
    ENUMERATE_CELL (icell, iface->cells()) {
      info() << "face cell " << icell.index() << " lid " << icell.localId() << " uid " << icell->uniqueId().asInt64();
    }
    ENUMERATE_EDGE (iedge, iface->edges()) {
      info() << "face edge " << iedge.index() << " lid " << iedge.localId() << " uid " << iedge->uniqueId().asInt64();
    }
  }
  // ALL NODES
  ENUMERATE_NODE (inode, mesh->allNodes()) {
    info() << "node with index " << inode.index();
    info() << "node with lid " << inode.localId();
    info() << "node with uid " << inode->uniqueId().asInt64();
    info() << "node number of faces " << inode->faces().size();
    info() << "node number of cells " << inode->cells().size();
    info() << "node number of edges " << inode->edges().size();
    ENUMERATE_FACE (iface, inode->faces()) {
      info() << "node face " << iface.index() << " lid " << iface.localId() << " uid " << iface->uniqueId().asInt64();
    }
    ENUMERATE_CELL (icell, inode->cells()) {
      info() << "node cell " << icell.index() << " lid " << icell.localId() << " uid " << icell->uniqueId().asInt64();
    }
    ENUMERATE_EDGE (iedge, inode->edges()) {
      info() << "face edge " << iedge.index() << " lid " << iedge.localId() << " uid " << iedge->uniqueId().asInt64();
    }
  }
  // ALL EDGES
  ENUMERATE_EDGE (iedge, mesh->allEdges()) {
    info() << "edge with index " << iedge.index();
    info() << "edge with lid " << iedge.localId();
    info() << "edge with uid " << iedge->uniqueId().asInt64();
    info() << "edge number of faces " << iedge->faces().size();
    info() << "edge number of cells " << iedge->cells().size();
    info() << "edge number of edges " << iedge->nodes().size();
    ENUMERATE_FACE (iface, iedge->faces()) {
      info() << "edge face " << iface.index() << " lid " << iface.localId() << " uid " << iface->uniqueId().asInt64();
    }
    ENUMERATE_CELL (icell, iedge->cells()) {
      info() << "edge cell " << icell.index() << " lid " << icell.localId() << " uid " << icell->uniqueId().asInt64();
    }
    ENUMERATE_NODE (inode, iedge->nodes()) {
      info() << "edge node " << inode.index() << " lid " << inode.localId() << " uid " << inode->uniqueId().asInt64();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CustomMeshTestModule::_testVariableAndGroups(IMesh* mesh)
{
  // test variables
  info() << " -- test variables -- ";
  // cell variable
  VariableCellInt32 cell_var{ VariableBuildInfo{ mesh, "cellvariable", mesh->cellFamily()->name() } };
  cell_var.fill(1);
  auto cell_var_sum = 0;
  ENUMERATE_CELL (icell,mesh->allCells()) {
    info() << "cell_variable at cell " << icell.localId() << " " << cell_var[icell];
    cell_var_sum += cell_var[icell];
  }
  if (cell_var_sum != mesh->allCells().size()) fatal() << "Error on cell variables";
  // node variable
  VariableNodeInt32 node_var{ VariableBuildInfo{ mesh, "nodevariable", mesh->nodeFamily()->name() } };
  node_var.fill(1);
  auto node_var_sum = 0;
  ENUMERATE_NODE(inode,mesh->allNodes()) {
    info() << "node_var at node " << inode.localId() << " " << node_var[inode];
    node_var_sum += node_var[inode];
  }
  if (node_var_sum != mesh->allNodes().size()) fatal() << "Error on node variables";
  // face variable
  VariableFaceInt32 face_var{ VariableBuildInfo{ mesh, "facevariable", mesh->faceFamily()->name() } };
  face_var.fill(1);
  auto face_var_sum = 0;
  ENUMERATE_FACE(iface,mesh->allFaces()) {
    info() << "face_var at face " << iface.localId() << " " << face_var[iface];
    face_var_sum += face_var[iface];
  }
  if (face_var_sum != mesh->allFaces().size()) fatal() << "Error on face variables";
  // edge variable
  VariableEdgeInt32 edge_var{ VariableBuildInfo{ mesh, "edgevariable", mesh->edgeFamily()->name() } };
  edge_var.fill(1);
  auto edge_var_sum = 0;
  ENUMERATE_EDGE(iedge,mesh->allEdges()) {
    info() << "edge_var at edge " << iedge.localId() << " " << edge_var[iedge];
    edge_var_sum += edge_var[iedge];
  }
  if (edge_var_sum != mesh->allEdges().size()) fatal() << "Error on edge variables";
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_CUSTOMMESHTEST(CustomMeshTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}// End namespace ArcaneTest::CustomMesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/