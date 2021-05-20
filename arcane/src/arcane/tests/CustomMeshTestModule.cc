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
      auto all_cells = mesh->allCells();
      ENUMERATE_CELL (icell,all_cells) {
        info() << "cell with index " << icell.index();
        info() << "cell with lid " << icell.localId();
        info() << "cell with uid " << icell->uniqueId().asInt64();
      }
    }
    else info() << "No Mesh";
//    ENUMERATE_CELL (icell, all_cells){
//      icell->localId();
//    }


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