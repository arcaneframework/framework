// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
    auto mesh = meshHandle().meshBase();
//    auto all_cells = mesh->allCells();
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