// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableInShMemUnitTest.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Module de test de l'AMR type 3.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MachineShMemWinVariable.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"

#include "arcane/tests/VariableInShMemUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableInShMemUnitTest
: public ArcaneVariableInShMemUnitTestObject
{

 public:

  explicit VariableInShMemUnitTest(const ModuleBuildInfo& mbi);
  ~VariableInShMemUnitTest() override = default;

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void init() override;
  void compute() override;
  void _adaptMesh();
  void _reset();
  void _test1();
  void _test2();
  void _test3();
  void _test10();
  void _test1_1();
  void _test1_2();
  void testMarkCellsToRefine(Integer level);

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableInShMemUnitTest::
VariableInShMemUnitTest(const ModuleBuildInfo& mbi)
: ArcaneVariableInShMemUnitTestObject(mbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("VariableInShMemUnitTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("VariableInShMemUnitTest.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("VariableInShMemUnitTest.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("VariableInShMemUnitTest");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
init()
{
  m_cartesian_mesh = ICartesianMesh::getReference(mesh());
  if (subDomain()->isContinue())
    m_cartesian_mesh->recreateFromDump();
  else {
    m_cartesian_mesh->computeDirections();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
compute()
{
  // _test1();
  // _reset();
  // _test2();
  // _reset();
  _test3();
  _reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_adaptMesh()
{
  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);

  amr_mng.beginAdaptMesh(2, 0);
  testMarkCellsToRefine(0);
  amr_mng.adaptLevel(0);
  amr_mng.endAdaptMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_reset()
{
  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);

  amr_mng.beginAdaptMesh(1, 0);
  amr_mng.endAdaptMesh();

  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test1()
{
  VariableCellInt32 var(VariableBuildInfo(mesh(), "Test1", IVariable::PInShMem));

  ENUMERATE_ (Cell, icell, ownCells()) {
    var[icell] = icell->uniqueId();
  }

  var.synchronize();

  ENUMERATE_ (Cell, icell, allCells()) {
    ARCANE_FATAL_IF(var[icell] != icell->uniqueId(),
                    "Error _test1() 1 : CellUID in variable invalid -- CellUID : {0} -- UIDInVar : {1}", icell->uniqueId(), var[icell]);
  }

  info() << "1 OK";

  _adaptMesh();

  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(0)) {
    ARCANE_FATAL_IF(var[icell] != icell->uniqueId(),
                    "Error _test1() 2 : CellUID in variable invalid -- CellUID : {0} -- UIDInVar : {1}", icell->uniqueId(), var[icell]);
  }

  info() << "2 OK";

  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(1)) {
    var[icell] = icell->uniqueId();
  }

  var.synchronize();

  ENUMERATE_ (Cell, icell, allCells()) {
    ARCANE_FATAL_IF(var[icell] != icell->uniqueId(),
                    "Error _test1() 3 : CellUID in variable invalid -- CellUID : {0} -- UIDInVar : {1}", icell->uniqueId(), var[icell]);
  }

  info() << "3 OK";

  _reset();
  var.synchronize();

  ENUMERATE_ (Cell, icell, allCells()) {
    ARCANE_FATAL_IF(var[icell] != icell->uniqueId(),
                    "Error _test1() 4 : CellUID in variable invalid -- CellUID : {0} -- UIDInVar : {1}", icell->uniqueId(), var[icell]);
  }
  info() << "4 OK";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test2()
{
  IParallelMng* pm = parallelMng();
  VariableCellInt32 var(VariableBuildInfo(mesh(), "Test2", IVariable::PInShMem));

  MachineShMemWinVariableItemT var_sh(var);

  ENUMERATE_ (Cell, icell, allCells()) {
    var[icell] = pm->commRank();
  }

  info() << "var.asArray().size(); : " << var.asArray().size();
  info() << "var_sh.segmentView().size(); : " << var_sh.segmentView().size();

  ConstArrayView<Int32> machine_ranks = var_sh.machineRanks();
  for (Int32 rank : machine_ranks) {
    Span<Int32> view = var_sh.segmentView(rank);
    for (const Int32 elem : view) {
      ARCANE_FATAL_IF(elem != rank,
                      "Error _test2() 1 : Invalid rank -- Expected : {0} -- Found : {1}", rank, elem);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test3()
{
  {
    VariableArrayInt32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinVariableArrayT var_sh(var);

    var.resize(2);
    var[0] = parallelMng()->commRank();
    var[1] = parallelMng()->commRank();

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value : " << var_sh.segmentView(rank);
    }
  }
  {
    VariableCellInt32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinVariableItemT var_sh(var);

    ENUMERATE_ (Cell, icell, allCells()) {
      var[icell] = parallelMng()->commRank();
    }

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value : " << var_sh.segmentView(rank);
    }
  }
  {
    VariableArray2Int32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinVariableArray2T var_sh(var);

    var.resize(2, 3);

    var(0, 0) = parallelMng()->commRank();
    var(0, 1) = parallelMng()->commRank();
    var(0, 2) = parallelMng()->commRank();
    var(1, 0) = parallelMng()->commRank()*10;
    var(1, 1) = parallelMng()->commRank()*10;
    var(1, 2) = parallelMng()->commRank()*10;

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value0 : " << var_sh.segmentView(rank)[0];
      info() << "Rank " << rank << " -- Value1 : " << var_sh.segmentView(rank)[1];
    }

  }
  {
    VariableCellArrayInt32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinVariableItemArrayT var_sh(var);

    var.resize(2);

    ENUMERATE_ (Cell, icell, allCells()) {
      var(icell, 0) = parallelMng()->commRank();
      var(icell, 1) = parallelMng()->commRank()*10;
    }

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value0 : " << var_sh.segmentView1D(rank);
      info() << "Rank " << rank << " -- Value1 : " << var_sh.segmentView1D(rank);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test10()
{
  Int32 properties = 0;
  if (parallelMng()->commRank() == 0) {
    properties = (IVariable::PInShMem | IVariable::PPersistant);
    //properties = (IVariable::PPersistant);
  }
  else {
    properties = (IVariable::PInShMem | IVariable::PPersistant | IVariable::PNoDump);
    //properties = (IVariable::PPersistant | IVariable::PNoDump);
  }

  VariableCellInt32 var(VariableBuildInfo(mesh(), "AAA", properties));
  VariableArrayInt32 var2(VariableBuildInfo(mesh(), "BBB", properties));

  MachineShMemWinVariableItemT var_sh(var);

  auto var_compute = [&]() -> void {
    debug() << "asArray().size() : " << var.asArray().size();
    auto ranks = var_sh.machineRanks();
    for (Int32 rank : ranks) {
      debug() << "Sizeof rank " << rank << " : "
              << var_sh.segmentView(rank).size();
    }
    debug() << "Array : " << var.asArray();

    ENUMERATE_ (Cell, icell, allCells()) {
      var[icell] = icell.localId();
    }
  };
  if (globalIteration() == 1) {
    var2.resize(10);
    for (Int32& a : var2) {
      a = parallelMng()->commRank();
    }
  }
  info() << "Array : " << var2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test1_1()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test1_2()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
testMarkCellsToRefine(Integer level)
{
  CartesianMeshNumberingMng numbering(m_cartesian_mesh);

  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(level)) {
    const Integer pos_x = numbering.offsetLevelToLevel(numbering.cellUniqueIdToCoordX(*icell), level, 0);
    const Integer pos_y = numbering.offsetLevelToLevel(numbering.cellUniqueIdToCoordY(*icell), level, 0);

    if (pos_x >= 2 && pos_x < 6 && pos_y >= 2 && pos_y < 5) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
    }
    if (pos_x >= 7 && pos_x < 11 && pos_y >= 6 && pos_y < 9) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_VARIABLEINSHMEMUNITTEST(VariableInShMemUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
