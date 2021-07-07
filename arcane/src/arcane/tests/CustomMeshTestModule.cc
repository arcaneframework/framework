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
      auto mesh = mesh_handle.mesh();
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
        ENUMERATE_NODE(inode, iedge->nodes()) {
          info() << "edge node " << inode.index() << " lid " << inode.localId() << " uid " << inode->uniqueId().asInt64();
        }
      }
    }
    else info() << "No Mesh";

    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_CUSTOMMESHTEST(CustomMeshTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}// End namespace ArcaneTest::CustomMesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/