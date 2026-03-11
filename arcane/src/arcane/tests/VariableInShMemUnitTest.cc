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
#include "arcane/core/MeshMDVariableRef.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"

#include "arcane/tests/VariableInShMemUnitTest_axl.h"
#include "arcane/utils/NumMatrix.h"
#include "arcane/utils/NumVector.h"
#include "arccore/base/MDDim.h"

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
  void _test4();
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
  if (!ParallelMngUtils::isMachineShMemWinAvailable(parallelMng())) {
    // Problème avec MPI. Peut intervenir si MPICH est compilé en mode ch3:sock.
    // On ne plante pas les tests dans ce cas.
    warning() << "Shared memory not supported";
    return;
  }
  _test1();
  _reset();
  _test2();
  _reset();
  _test3();
  _reset();
  _test4();
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

  MachineShMemWinMeshVariableScalarT var_sh(var);

  ENUMERATE_ (Cell, icell, allCells()) {
    var[icell] = pm->commRank();
  }

  var_sh.barrier();

  info() << "var.asArray().size(); : " << var.asArray().size();
  info() << "var_sh.segmentView().size(); : " << var_sh.view(pm->commRank()).size();
  info() << "var_sh(); : " << var_sh(pm->commRank(), 0);

  ConstArrayView<Int32> machine_ranks = var_sh.machineRanks();
  for (Int32 rank : machine_ranks) {
    Span<Int32> view = var_sh.view(rank);
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
    var_sh.updateVariable();

    var[0] = parallelMng()->commRank();
    var[1] = parallelMng()->commRank();

    var_sh.barrier();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value : " << var_sh.view(rank);
    }
  }
  {
    VariableCellInt32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinMeshVariableScalarT var_sh(var);

    ENUMERATE_ (Cell, icell, allCells()) {
      var[icell] = parallelMng()->commRank();
    }

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- Value : " << var_sh.view(rank);
      info() << "Rank " << rank << " -- Value : " << var_sh(rank, 0);
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
      info() << "Rank " << rank << " -- Value0 : " << var_sh.view(rank)[0];
      info() << "Rank " << rank << " -- Value1 : " << var_sh.view(rank)[1];
    }

  }
  {
    VariableCellArrayInt32 var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinMeshVariableArrayT var_sh(var);

    var.resize(2);

    ENUMERATE_ (Cell, icell, allCells()) {
      var(icell, 0) = parallelMng()->commRank();
      var(icell, 1) = parallelMng()->commRank()*10;
    }

    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();

    for (Int32 rank : machine_ranks) {
      info() << "Rank " << rank << " -- ValuesCell0 : " << var_sh.view(rank)(0);
      info() << "Rank " << rank << " -- ValuesCell1 : " << var_sh.view(rank)(1);
    }
  }
  {
    MeshMDVariableRefT<Cell, Real, MDDim2> var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));

    MachineShMemWinMeshMDVariableT var_sh(var);

    var.reshape({ 2, 3 });

    ENUMERATE_ (Cell, icell, allCells()) {
      var(icell, 0, 0) = parallelMng()->commRank();
      var(icell, 0, 1) = parallelMng()->commRank() * 10;
      var(icell, 0, 2) = parallelMng()->commRank() * 20;
      var(icell, 1, 0) = parallelMng()->commRank() * 30;
      var(icell, 1, 1) = parallelMng()->commRank() * 40;
      var(icell, 1, 2) = parallelMng()->commRank() * 50;
    }
    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();
    for (Int32 rank : machine_ranks) {
      MDSpan<Real, MDDim3> aaa = var_sh.view(rank);
      info() << "Rank " << rank << " -- aaa.extents().extent0() : " << aaa.extent0();
      info() << "Rank " << rank << " -- aaa.extents().extent1() : " << aaa.extent1();
      info() << "Rank " << rank << " -- aaa.extents().extent2() : " << aaa.extent2();
      // info() << "Rank " << rank << " -- Value : " << aaa.to1DSpan();
    }

    // MDSpan<Real, MDDim2> bbb = var_sh(0, 0);
  }
  {
    MeshVectorMDVariableRefT<Cell, Real, 3, MDDim2> var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinMeshVectorMDVariableT var_sh(var);

    var.reshape({ 2, 3 });

    ENUMERATE_ (Cell, icell, allCells()) {
      var(icell, 0, 0) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank()), 0, 0 };
      var(icell, 0, 1) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank() * 10), 0, 0 };
      var(icell, 0, 2) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank() * 20), 0, 0 };
      var(icell, 1, 0) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank() * 30), 0, 0 };
      var(icell, 1, 1) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank() * 40), 0, 0 };
      var(icell, 1, 2) = NumVector<Real, 3>{ static_cast<Real>(parallelMng()->commRank() * 50), 0, 0 };
    }
    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();
    for (Int32 rank : machine_ranks) {
      MDSpan<Real, MDDim4> aaa = var_sh.view(rank);
      info() << "Rank " << rank << " -- aaa.extents().extent0() : " << aaa.extent0();
      info() << "Rank " << rank << " -- aaa.extents().extent1() : " << aaa.extent1();
      info() << "Rank " << rank << " -- aaa.extents().extent2() : " << aaa.extent2();
      info() << "Rank " << rank << " -- aaa.extents().extent3() : " << aaa.extent3();
      // info() << "Rank " << rank << " -- Value : " << aaa.to1DSpan();
    }
  }
  {
    MeshMatrixMDVariableRefT<Cell, Real, 2, 4, MDDim1> var(VariableBuildInfo(mesh(), "Test3", IVariable::PInShMem));
    MachineShMemWinMeshMatrixMDVariableT var_sh(var);

    var.reshape({ 3 });

    ENUMERATE_ (Cell, icell, allCells()) {
      var(icell, 0) = Arcane::NumMatrix<Real, 2, 4>{ { static_cast<Real>(parallelMng()->commRank()), 0, 0, 0 }, { static_cast<Real>(parallelMng()->commRank()), 0, 0, 0 } };
      var(icell, 1) = Arcane::NumMatrix<Real, 2, 4>{ { static_cast<Real>(parallelMng()->commRank() * 10), 0, 0, 0 }, { static_cast<Real>(parallelMng()->commRank() * 10), 0, 0, 0 } };
      var(icell, 2) = Arcane::NumMatrix<Real, 2, 4>{ { static_cast<Real>(parallelMng()->commRank() * 20), 0, 0, 0 }, { static_cast<Real>(parallelMng()->commRank() * 20), 0, 0, 0 } };
    }
    var_sh.updateVariable();

    auto machine_ranks = var_sh.machineRanks();
    for (Int32 rank : machine_ranks) {
      MDSpan<Real, MDDim4> aaa = var_sh.view(rank);
      info() << "Rank " << rank << " -- aaa.extents().extent0() : " << aaa.extent0();
      info() << "Rank " << rank << " -- aaa.extents().extent1() : " << aaa.extent1();
      info() << "Rank " << rank << " -- aaa.extents().extent2() : " << aaa.extent2();
      info() << "Rank " << rank << " -- aaa.extents().extent3() : " << aaa.extent3();
      // info() << "Rank " << rank << " -- Value : " << aaa.to1DSpan();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInShMemUnitTest::
_test4()
{
  Int32 properties = 0;
  if (parallelMng()->commRank() == 0) {
    properties = (IVariable::PInShMem | IVariable::PPersistant);
  }
  else {
    properties = (IVariable::PInShMem | IVariable::PPersistant | IVariable::PDumpNull);
  }

  VariableArrayInt32 var2(VariableBuildInfo(mesh(), "BBB", properties));

  if (globalIteration() == 1) {
    var2.resize(10);
    for (Int32& a : var2) {
      a = parallelMng()->commRank();
    }
  }
  else if (subDomain()->isContinue()) {
    if (parallelMng()->commRank() == 0) {
      ARCANE_FATAL_IF(var2.size() != 10,
                      "Error _test4() 1 : Array size is invalid -- Expected : 10 -- Found : {0}", var2.size());
    }
    else {
      ARCANE_FATAL_IF(!var2.empty(),
                      "Error _test4() 1 : Array size is invalid -- Expected : 0 -- Found : {0}", var2.size());
    }
  }
  info() << "Array : " << var2;
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
