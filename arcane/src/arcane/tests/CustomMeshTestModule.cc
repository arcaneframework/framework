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

#include "arcane/ItemGroup.h"
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
      _testVariables(mesh_handle.mesh());
      _testGroups(mesh_handle.mesh());
    }
    else info() << "No Mesh";

    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }

 private:
  void _testEnumerationAndConnectivities(IMesh* mesh);
  void _testVariables(IMesh* mesh);
  void _testGroups(IMesh* mesh);
  void _buildGroup(IItemFamily* family, String const& group_name);
  template <typename VariableRefType>
  void _checkVariable(VariableRefType variable, ItemGroup item_group);
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

void CustomMeshTestModule::_testVariables(IMesh* mesh)
{
  // test variables
  info() << " -- test variables -- ";
  // cell variable
  VariableCellInt32 cell_var{ VariableBuildInfo{ mesh, "cellvariable", mesh->cellFamily()->name() } };
  _checkVariable(cell_var, mesh->allCells());
  // node variable
  VariableNodeInt32 node_var{ VariableBuildInfo{ mesh, "nodevariable", mesh->nodeFamily()->name() } };
  _checkVariable(node_var, mesh->allNodes());
  // face variable
  VariableFaceInt32 face_var{ VariableBuildInfo{ mesh, "facevariable", mesh->faceFamily()->name() } };
  _checkVariable(face_var, mesh->allFaces());
  // edge variable
  VariableEdgeInt32 edge_var{ VariableBuildInfo{ mesh, "edgevariable", mesh->edgeFamily()->name() } };
  _checkVariable(edge_var, mesh->allEdges());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CustomMeshTestModule::
_testGroups(IMesh* mesh)
{
  // Cell group
  String group_name = "my_cell_group";
  _buildGroup(mesh->cellFamily(),group_name);
  PartialVariableCellInt32 partial_cell_var({mesh, "partial_cell_variable", mesh->cellFamily()->name(), group_name});
  _checkVariable(partial_cell_var, partial_cell_var.itemGroup());
  // Node group
  group_name = "my_node_group";
  _buildGroup(mesh->nodeFamily(),group_name);
  PartialVariableNodeInt32 partial_node_var({mesh, "partial_node_variable", mesh->nodeFamily()->name(), group_name});
  _checkVariable(partial_node_var, partial_node_var.itemGroup());
  // Face group
  group_name = "my_face_group";
  _buildGroup(mesh->faceFamily(),group_name);
  PartialVariableFaceInt32 partial_face_var({mesh, "partial_face_variable", mesh->faceFamily()->name(), group_name});
  _checkVariable(partial_face_var, partial_face_var.itemGroup());
  // Edge group
  group_name = "my_edge_group";
  _buildGroup(mesh->edgeFamily(),group_name);
  PartialVariableEdgeInt32 partial_edge_var({mesh, "partial_edge_variable", mesh->edgeFamily()->name(), group_name});
  _checkVariable(partial_edge_var, partial_edge_var.itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CustomMeshTestModule::
_buildGroup(IItemFamily* family, String const& group_name)
{
  auto group = family->findGroup(group_name,true);
  Int32UniqueArray item_lids;
  item_lids.reserve(family->nbItem());
  ENUMERATE_ITEM (iitem, family->allItems()) {
    if (iitem.localId()%2 ==0)
      item_lids.add(iitem.localId());
  }
  group.addItems(item_lids);
  info() << itemKindName(family->itemKind()) <<" group size " << group.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename VariableRefType>
void CustomMeshTestModule::
_checkVariable(VariableRefType variable_ref, ItemGroup item_group)
{
  variable_ref.fill(1);
  auto variable_sum = 0;
  A_ENUMERATE_ITEM (ItemEnumeratorT<typename VariableRefType::ItemType>,iitem, item_group) {
    info() << variable_ref.name() << " at item " << iitem.localId() << " " << variable_ref[iitem];
    variable_sum += variable_ref[iitem];
  }
  if (variable_sum != item_group.size()) fatal() << "Error on variable " << variable_ref.name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_CUSTOMMESHTEST(CustomMeshTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}// End namespace ArcaneTest::CustomMesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/