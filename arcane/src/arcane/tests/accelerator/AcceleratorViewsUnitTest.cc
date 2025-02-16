// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorViewsUnitTest.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Service de test des vues pour les accelerateurs.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/MultiArray2.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandEnumerate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'AcceleratorViews'.
 */
class AcceleratorViewsUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorViewsUnitTest(const ServiceBuildInfo& cb);
  ~AcceleratorViewsUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;
  ax::RunQueue m_queue;
  VariableCellArrayReal m_cell_array1;
  VariableCellArrayReal m_cell_array2;
  VariableCellReal2 m_cell1_real2;
  VariableCellReal3 m_cell1_real3;
  VariableCellReal2x2 m_cell1_real2x2;
  VariableCellReal3x3 m_cell1_real3x3;
  PartialVariableCellReal m_partial_cell1;
  PartialVariableCellReal m_partial_cell2;
  PartialVariableCellReal3 m_partial_cell1_real3;
  PartialVariableCellReal3 m_partial_cell2_real3;
  PartialVariableCellArrayReal m_partial_cell_array1;
  PartialVariableCellArrayReal m_partial_cell_array2;

 private:

  void _setCellArrayValue(Integer seed);
  void _checkCellArrayValue(const String& message) const;

 public:

  void _executeTest1();
  void _executeTest2();
  void _executeTest3();
  void _executeTestReal2x2();
  void _executeTestReal3x3();
  void _executeTest2Real3x3();
  void _executeTestMemoryCopy();
  void _executeTestVariableCopy();
  void _executeTestVariableFill();
  void _checkResultReal2(Real to_add);
  void _checkResultReal3(Real to_add);
  void _checkResultReal2x2(Real to_add);
  void _checkResultReal3x3(Real to_add);
  void _executeTestGroupIndexTable();
  void _executeTestMultiArray();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorViewsUnitTest, IUnitTest,
                                           AcceleratorViewsUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorViewsUnitTest::
AcceleratorViewsUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_cell_array1(VariableBuildInfo(sb.mesh(), "CellArray1"))
, m_cell_array2(VariableBuildInfo(sb.mesh(), "CellArray2"))
, m_cell1_real2(VariableBuildInfo(sb.mesh(), "Cell1Real2"))
, m_cell1_real3(VariableBuildInfo(sb.mesh(), "Cell1Real3"))
, m_cell1_real2x2(VariableBuildInfo(sb.mesh(), "Cell1Real2x2"))
, m_cell1_real3x3(VariableBuildInfo(sb.mesh(), "Cell1Real3x3"))
, m_partial_cell1(VariableBuildInfo(sb.mesh(), "PartialCell1", "Cell", "MyPartialGroup"))
, m_partial_cell2(VariableBuildInfo(sb.mesh(), "PartialCell2", "Cell", "MyPartialGroup"))
, m_partial_cell1_real3(VariableBuildInfo(sb.mesh(), "PartialCell1Real3", "Cell", "MyPartialGroup"))
, m_partial_cell2_real3(VariableBuildInfo(sb.mesh(), "PartialCell2Real3", "Cell", "MyPartialGroup"))
, m_partial_cell_array1(VariableBuildInfo(sb.mesh(), "PartialCellArray1", "Cell", "MyPartialGroup"))
, m_partial_cell_array2(VariableBuildInfo(sb.mesh(), "PartialCellArray2", "Cell", "MyPartialGroup"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorViewsUnitTest::
~AcceleratorViewsUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
initializeTest()
{
  m_runner = subDomain()->acceleratorMng()->runner();
  m_queue = subDomain()->acceleratorMng()->queue();

  m_cell_array1.resize(12);
  m_cell_array2.resize(12);

  {
    UniqueArray<Int32> cells_to_add;
    ENUMERATE_ (Cell, icell, allCells()) {
      if ((icell.index() % 3) == 0)
        cells_to_add.add(icell.itemLocalId());
    }
    CellGroup pcells = m_partial_cell_array1.itemGroup();
    pcells.addItems(cells_to_add);
    const Int32 n = 12;
    m_partial_cell_array1.resize(n);
    m_partial_cell_array2.resize(n);
    ENUMERATE_ (Cell, icell, pcells) {
      ItemEnumeratorIndexT<Cell> x(icell.index());
      for (Int32 i = 0; i < n; ++i)
        m_partial_cell_array1[x][i] = static_cast<Real>(icell.itemLocalId() + icell.index() * 2);
      m_partial_cell1[x] = static_cast<Real>(icell.index()) * 2.5;
      m_partial_cell1_real3[x] = Real3(m_partial_cell1[x], m_partial_cell1[x] + 1.0, m_partial_cell1[x] + 2.0);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_setCellArrayValue(Integer seed)
{
  Integer n = m_cell_array1.arraySize();
  ENUMERATE_CELL (icell, allCells()) {
    Int32 lid = icell.itemLocalId();
    for (Integer i = 0; i < n; ++i) {
      m_cell_array1[icell][i] = Real((i + 1) + lid + seed);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_checkCellArrayValue(const String& message) const
{
  ValueChecker vc(A_FUNCINFO);
  Integer n = m_cell_array1.arraySize();
  info() << "Check CELL ARRAY VALUE n=" << n << " message=" << message;
  ENUMERATE_CELL (icell, allCells()) {
    vc.areEqual(m_cell_array1[icell], m_cell_array2[icell], message);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
executeTest()
{
  _executeTestGroupIndexTable();
  _executeTest1();
  _executeTest2();
  _executeTest3();
  _executeTestReal2x2();
  _executeTestReal3x3();
  _executeTest2Real3x3();
  _executeTestMemoryCopy();
  _executeTestVariableCopy();
  _executeTestVariableFill();
  _executeTestMultiArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest1()
{
  info() << "Test1";
  auto queue = makeQueue(m_runner);

  Integer dim2_size = m_cell_array1.arraySize();

  {
    int seed = 37;
    _setCellArrayValue(seed);

    {
      auto command = makeCommand(m_queue);
      auto in_cell_array1 = viewIn(command, m_cell_array1);
      auto out_cell_array2 = viewOut(command, m_cell_array2);

      command << RUNCOMMAND_ENUMERATE (CellLocalId, vi, allCells())
      {
        out_cell_array2[vi].copy(in_cell_array1[vi]);
      };
    }

    _checkCellArrayValue("View1");

    {
      auto command = makeCommand(m_queue);
      auto in_cell_array1 = viewIn(command, m_cell_array1);
      auto in_partial_cell_array1 = viewIn(command, m_partial_cell_array1);
      auto out_partial_cell_array2 = viewOut(command, m_partial_cell_array2);
      auto inout_partial_cell_array2 = viewInOut(command, m_partial_cell_array2);

      auto in_partial_cell1 = viewIn(command, m_partial_cell1);
      auto out_partial_cell2 = viewOut(command, m_partial_cell2);
      auto inout_partial_cell2 = viewOut(command, m_partial_cell2);

      auto in_partial_cell1_real3 = viewIn(command, m_partial_cell1_real3);
      auto out_partial_cell2_real3 = viewOut(command, m_partial_cell2_real3);
      auto inout_partial_cell2_real3 = viewInOut(command, m_partial_cell2_real3);
      command << RUNCOMMAND_ENUMERATE (IteratorWithIndex<CellLocalId>, vi, m_partial_cell_array1.itemGroup())
      {
        CellEnumeratorIndex iter_index(vi.index());
        CellLocalId cell_lid(vi.value());
        out_partial_cell_array2[iter_index].copy(in_cell_array1[cell_lid]);
        out_partial_cell_array2[iter_index][0] = in_partial_cell_array1[cell_lid][1];
        Real3 xyz(in_partial_cell1_real3[iter_index].y, in_partial_cell1_real3[iter_index].z, in_partial_cell1_real3[iter_index].x);
        Int32 modulo = vi.index() % 4;
        if (modulo == 3) {
          out_partial_cell_array2[iter_index][1] = in_partial_cell1[iter_index];
          out_partial_cell2[iter_index] = in_partial_cell1[iter_index];
          out_partial_cell2_real3[iter_index] = xyz;
        }
        else if (modulo == 2) {
          inout_partial_cell_array2[cell_lid][1] = in_partial_cell1[cell_lid];
          inout_partial_cell2[cell_lid] = in_partial_cell1[iter_index];
          inout_partial_cell2_real3[cell_lid] = xyz;
        }
        else if (modulo == 1) {
          out_partial_cell_array2[cell_lid][1] = in_partial_cell1[iter_index];
          out_partial_cell2[cell_lid] = in_partial_cell1[iter_index];
          out_partial_cell2_real3[cell_lid] = xyz;
        }
        else {
          inout_partial_cell_array2[iter_index][1] = in_partial_cell1[cell_lid];
          inout_partial_cell2[iter_index] = in_partial_cell1[iter_index];
          inout_partial_cell2_real3[iter_index] = xyz;
        }
      };
    }
    info() << "Check Partial values";
    ENUMERATE_ (Cell, iter, m_partial_cell_array1.itemGroup()) {
      CellEnumeratorIndex iter_index(iter.index());
      if (m_partial_cell2[iter_index] != m_partial_cell1[iter_index])
        ARCANE_FATAL("Bad value (1) iter={0} v1={1} v2={2}", iter.index(), m_partial_cell2[iter_index], m_partial_cell1[iter_index]);
      Real3 xyz(m_partial_cell1_real3[iter_index].y, m_partial_cell1_real3[iter_index].z, m_partial_cell1_real3[iter_index].x);
      if (xyz != m_partial_cell2_real3[iter_index])
        ARCANE_FATAL("Bad value (2) iter={0} v1={1} v2={2}", iter.index(), m_partial_cell2_real3[iter_index], xyz);
    }
  }

  {
    int seed = 23;
    _setCellArrayValue(seed);

    auto command = makeCommand(m_queue);

    auto in_cell_array1 = viewIn(command, m_cell_array1);
    auto out_cell_array2 = viewOut(command, m_cell_array2);

    command << RUNCOMMAND_ENUMERATE (CellLocalId, vi, allCells())
    {
      for (Integer i = 0; i < dim2_size; ++i)
        out_cell_array2[vi][i] = in_cell_array1[vi][i];
    };
    _checkCellArrayValue("View2");
  }

  {
    int seed = 53;
    _setCellArrayValue(seed);

    auto command = makeCommand(m_queue);

    auto in_cell_array1 = viewInOut(command, m_cell_array1);
    auto out_cell_array2 = viewOut(command, m_cell_array2);

    command << RUNCOMMAND_ENUMERATE (CellLocalId, vi, allCells())
    {
      out_cell_array2[vi].copy(in_cell_array1[vi]);
    };

    _checkCellArrayValue("View3");
  }

  {
    int seed = 93;
    _setCellArrayValue(seed);

    auto command = makeCommand(m_queue);

    auto in_cell_array1 = ax::viewIn(command, m_cell_array1);
    auto out_cell_array2 = ax::viewInOut(command, m_cell_array2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      out_cell_array2[vi].copy(in_cell_array1[vi]);
    };

    _checkCellArrayValue("View4");
  }

  {
    int seed = 43;
    _setCellArrayValue(seed);

    auto command = makeCommand(m_queue);

    auto inout_cell_array1 = ax::viewInOut(command, m_cell_array1);
    auto out_cell_array2 = ax::viewInOut(command, m_cell_array2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      out_cell_array2[vi].copy(inout_cell_array1[vi]);
    };

    _checkCellArrayValue("View5");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest2()
{
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2 = viewInOut(command, m_cell1_real2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2 ref_v;
      ref_v[0] = 2.0 + v;
      ref_v[1] = 3.0 + v;
      inout_cell1_real2[vi].setX(ref_v[0]);
      inout_cell1_real2[vi].setY(ref_v[1]);
    };
  }
  _checkResultReal2(2.0);

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2 = ax::viewInOut(command, m_cell1_real2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2 ref_v;
      ref_v[0] = 4.0 + v;
      ref_v[1] = 5.0 + v;
      inout_cell1_real2[vi][0] = ref_v[0];
      inout_cell1_real2[vi][1] = ref_v[1];
    };
  }
  _checkResultReal2(4.0);

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3 = ax::viewInOut(command, m_cell1_real3);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3 ref_v;
      ref_v[0] = 2.0 + v;
      ref_v[1] = 3.0 + v;
      ref_v[2] = 4.0 + v;
      inout_cell1_real3[vi].setX(ref_v[0]);
      inout_cell1_real3[vi].setY(ref_v[1]);
      inout_cell1_real3[vi].setZ(ref_v[2]);
    };
  }
  _checkResultReal3(2.0);

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3 = ax::viewInOut(command, m_cell1_real3);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3 ref_v;
      ref_v[0] = 6.0 + v;
      ref_v[1] = 7.0 + v;
      ref_v[2] = 8.0 + v;
      inout_cell1_real3[vi][0] = ref_v[0];
      inout_cell1_real3[vi][1] = ref_v[1];
      inout_cell1_real3[vi][2] = ref_v[2];
    };
  }
  _checkResultReal3(6.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest3()
{
  info() << "ExecuteTest3";
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);

  CellGroup own_cells = allCells().own();
  Int32 max_local_id = own_cells.itemFamily()->maxLocalId();
  NumArray<Int32, MDDim1> checked_local_ids(max_local_id);
  checked_local_ids.fill(-1, queue);
  {
    // Remplit out_checked_local_ids avec le i-ème localId() du groupe.
    auto out_checked_local_ids = ax::viewOut(command, checked_local_ids);
    command << RUNCOMMAND_ENUMERATE (IteratorWithIndex<CellLocalId>, vi, own_cells)
    {
      out_checked_local_ids[vi.index()] = vi.value();
    };
    // Vérifie tout est OK.
    ENUMERATE_ (Cell, icell, own_cells) {
      Int32 my_lid = icell.itemLocalId();
      Int32 computed_lid = checked_local_ids[icell.index()];
      if (my_lid != computed_lid)
        ARCANE_FATAL("Bad computed local id ref={0} computed={1}", my_lid, computed_lid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestReal2x2()
{
  info() << "Execute Test Real2x2";
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2x2 = ax::viewInOut(command, m_cell1_real2x2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2x2 ref_v;
      ref_v[0][0] = 2.0 + v;
      ref_v[1][0] = 3.0 + v;
      ref_v[0][1] = 4.0 + v;
      ref_v[1][1] = 5.0 + v;
      inout_cell1_real2x2[vi].setXX(ref_v[0][0]);
      inout_cell1_real2x2[vi].setYX(ref_v[1][0]);
      inout_cell1_real2x2[vi].setXY(ref_v[0][1]);
      inout_cell1_real2x2[vi].setYY(ref_v[1][1]);
    };
  }
  _checkResultReal2x2(2.0);

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2x2 = ax::viewInOut(command, m_cell1_real2x2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2x2 ref_v;
      ref_v[0][0] = 2.0 + v;
      ref_v[1][0] = 3.0 + v;
      ref_v[0][1] = 4.0 + v;
      ref_v[1][1] = 5.0 + v;
      inout_cell1_real2x2[vi][0][0] = ref_v[0][0];
      inout_cell1_real2x2[vi][1][0] = ref_v[1][0];
      inout_cell1_real2x2[vi][0][1] = ref_v[0][1];
      inout_cell1_real2x2[vi][1][1] = ref_v[1][1];
    };
  }
  _checkResultReal2x2(2.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestReal3x3()
{
  info() << "Execute Test Real3x3";
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3x3 = ax::viewInOut(command, m_cell1_real3x3);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3x3 ref_v;
      ref_v[0][0] = 2.0 + v;
      ref_v[1][0] = 3.0 + v;
      ref_v[2][0] = 4.0 + v;

      ref_v[0][1] = 5.0 + v;
      ref_v[1][1] = 6.0 + v;
      ref_v[2][1] = 7.0 + v;

      ref_v[0][2] = 8.0 + v;
      ref_v[1][2] = 9.0 + v;
      ref_v[2][2] = 10.0 + v;

      inout_cell1_real3x3[vi].setXX(ref_v[0][0]);
      inout_cell1_real3x3[vi].setYX(ref_v[1][0]);
      inout_cell1_real3x3[vi].setZX(ref_v[2][0]);

      inout_cell1_real3x3[vi].setXY(ref_v[0][1]);
      inout_cell1_real3x3[vi].setYY(ref_v[1][1]);
      inout_cell1_real3x3[vi].setZY(ref_v[2][1]);

      inout_cell1_real3x3[vi].setXZ(ref_v[0][2]);
      inout_cell1_real3x3[vi].setYZ(ref_v[1][2]);
      inout_cell1_real3x3[vi].setZZ(ref_v[2][2]);
    };
  }
  _checkResultReal3x3(2.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest2Real3x3()
{
  info() << "Execute Test2 Real3x3";
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3x3 = ax::viewInOut(command, m_cell1_real3x3);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3x3 ref_v;
      ref_v[0][0] = 5.0 + v;
      ref_v[1][0] = 6.0 + v;
      ref_v[2][0] = 7.0 + v;

      ref_v[0][1] = 8.0 + v;
      ref_v[1][1] = 9.0 + v;
      ref_v[2][1] = 10.0 + v;

      ref_v[0][2] = 11.0 + v;
      ref_v[1][2] = 12.0 + v;
      ref_v[2][2] = 13.0 + v;

      inout_cell1_real3x3[vi].setX(ref_v[0]);
      inout_cell1_real3x3[vi].setY(ref_v[1]);
      inout_cell1_real3x3[vi].setZ(ref_v[2]);
    };
  }
  _checkResultReal3x3(5.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_checkResultReal2(Real to_add)
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ (Cell, vi, allCells()) {
    Real vbase = static_cast<Real>(vi.itemLocalId());
    Real v = to_add + vbase;
    Real rx = 0.0 + v;
    Real ry = 1.0 + v;
    vc.areEqual(rx, m_cell1_real2[vi].x, "CheckX");
    vc.areEqual(ry, m_cell1_real2[vi].y, "CheckY");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_checkResultReal3(Real to_add)
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ (Cell, vi, allCells()) {
    Real vbase = static_cast<Real>(vi.itemLocalId());
    Real v = to_add + vbase;
    Real rx = 0.0 + v;
    Real ry = 1.0 + v;
    Real rz = 2.0 + v;
    vc.areEqual(rx, m_cell1_real3[vi].x, "CheckX");
    vc.areEqual(ry, m_cell1_real3[vi].y, "CheckY");
    vc.areEqual(rz, m_cell1_real3[vi].z, "CheckZ");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_checkResultReal2x2(Real to_add)
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ (Cell, vi, allCells()) {
    Real vbase = static_cast<Real>(vi.itemLocalId());
    Real v = to_add + vbase;
    Real rxx = 0.0 + v;
    Real ryx = 1.0 + v;
    Real rxy = 2.0 + v;
    Real ryy = 3.0 + v;
    vc.areEqual(rxx, m_cell1_real2x2[vi].x.x, "Real2x2CheckXX");
    vc.areEqual(rxy, m_cell1_real2x2[vi].x.y, "Real2x2CheckXY");
    vc.areEqual(ryx, m_cell1_real2x2[vi].y.x, "Real2x2CheckYX");
    vc.areEqual(ryy, m_cell1_real2x2[vi].y.y, "Real2x2CheckYY");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_checkResultReal3x3(Real to_add)
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ (Cell, vi, allCells()) {
    Real vbase = static_cast<Real>(vi.itemLocalId());
    Real v = to_add + vbase;
    Real rxx = 0.0 + v;
    Real ryx = 1.0 + v;
    Real rzx = 2.0 + v;
    Real rxy = 3.0 + v;
    Real ryy = 4.0 + v;
    Real rzy = 5.0 + v;
    Real rxz = 6.0 + v;
    Real ryz = 7.0 + v;
    Real rzz = 8.0 + v;
    vc.areEqual(rxx, m_cell1_real3x3[vi].x.x, "Real3x3CheckXX");
    vc.areEqual(ryx, m_cell1_real3x3[vi].y.x, "Real3x3CheckYX");
    vc.areEqual(rzx, m_cell1_real3x3[vi].z.x, "Real3x3CheckZX");

    vc.areEqual(rxy, m_cell1_real3x3[vi].x.y, "Real3x3CheckXY");
    vc.areEqual(ryy, m_cell1_real3x3[vi].y.y, "Real3x3CheckYY");
    vc.areEqual(rzy, m_cell1_real3x3[vi].z.y, "Real3x3CheckZY");

    vc.areEqual(rxz, m_cell1_real3x3[vi].x.z, "Real3x3CheckXZ");
    vc.areEqual(ryz, m_cell1_real3x3[vi].y.z, "Real3x3CheckYZ");
    vc.areEqual(rzz, m_cell1_real3x3[vi].z.z, "Real3x3CheckZZ");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestMemoryCopy()
{
  info() << "Execute Test MemoryCopy";
  eMemoryRessource source_mem = eMemoryRessource::Host;
  eMemoryRessource dest_mem = eMemoryRessource::Host;
  if (ax::impl::isAcceleratorPolicy(m_runner.executionPolicy()))
    dest_mem = eMemoryRessource::Device;

  const int nb_value = 100000;
  NumArray<Int32, MDDim1> a(nb_value, source_mem);
  NumArray<Int32, MDDim1> b(nb_value, source_mem);
  NumArray<Int32, MDDim1> c(nb_value, source_mem);

  // Initialise les tableaux
  for (int i = 0; i < nb_value; ++i) {
    a(i) = i + 3;
    b[i] = i + 5;
  }

  NumArray<Int32, MDDim1> d_a(nb_value, dest_mem);
  NumArray<Int32, MDDim1> d_b(nb_value, dest_mem);
  NumArray<Int32, MDDim1> d_c(nb_value, dest_mem);

  // Copie explicitement les données dans le device
  // Test la construction en donnant les tailles explicitement.
  auto queue = makeQueue(m_runner);
  queue.copyMemory(ax::MemoryCopyArgs(d_a.bytes().data(), a.bytes().data(), nb_value * sizeof(Int32)).addAsync());
  queue.copyMemory(ax::MemoryCopyArgs(d_b.bytes().data(), b.bytes().data(), nb_value * sizeof(Int32)).addAsync());

  {
    auto command = makeCommand(queue);
    auto in_a = viewIn(command, d_a);
    auto in_b = viewIn(command, d_b);
    auto out_c = viewOut(command, d_c);

    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      out_c(iter) = in_a[iter] + in_b(iter);
    };
  }
  // Recopie du device vers l'hôte
  queue.copyMemory(ax::MemoryCopyArgs(MutableMemoryView(c.bytes()), ConstMemoryView(d_c.bytes())));

  // Vérifie que tout est OK.
  // Initialise les tableaux
  for (int i = 0; i < nb_value; ++i) {
    Int32 expected_value = (i + 3) + (i + 5);
    if (c(i) != expected_value)
      ARCANE_FATAL("Bad value index={0} value={1} expected={2}", i, c(i), expected_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestVariableCopy()
{
  info() << "Execute Test VariableCopy";

  ValueChecker vc(A_FUNCINFO);

  auto queue = makeQueue(m_runner).addAsync(true);
  if (!queue.isAsync())
    ARCANE_FATAL("Queue is not asynchronous");

  VariableCellReal2 test_cell1_real2(VariableBuildInfo(mesh(), "TestCell1Real2"));
  VariableCellReal3 test_cell1_real3(VariableBuildInfo(mesh(), "TestCell1Real3"));
  VariableCellReal2x2 test_cell1_real2x2(VariableBuildInfo(mesh(), "TestCell1Real2x2"));
  VariableCellReal3x3 test_cell1_real3x3(VariableBuildInfo(mesh(), "TestCell1Real3x3"));
  VariableCellArrayReal test_cell1_array(VariableBuildInfo(mesh(), "TestCellArrayReal2"));
  VariableCellArrayReal test_cell2_array(VariableBuildInfo(mesh(), "TestCellArrayReal3"));
  VariableCellReal2 test2_cell1_real2(VariableBuildInfo(mesh(), "Test2Cell1Real2"));

  test_cell1_array.resize(m_cell_array1.arraySize());
  test_cell2_array.resize(m_cell_array2.arraySize());

  test_cell1_real2.copy(m_cell1_real2, &queue);
  test_cell1_real3.copy(m_cell1_real3, &queue);
  test_cell1_real2x2.copy(m_cell1_real2x2, &queue);
  test_cell1_real3x3.copy(m_cell1_real3x3, &queue);
  test2_cell1_real2.copy(m_cell1_real2, nullptr);
  test_cell1_array.copy(m_cell_array1, &queue);
  test_cell2_array.copy(m_cell_array2, nullptr);
  queue.barrier();

  vc.areEqualArray(test_cell1_real2._internalSpan(), m_cell1_real2._internalSpan(), "CompareReal2");
  vc.areEqualArray(test_cell1_real3._internalSpan(), m_cell1_real3._internalSpan(), "CompareReal3");
  vc.areEqualArray(test_cell1_real2x2._internalSpan(), m_cell1_real2x2._internalSpan(), "CompareReal2x2");
  vc.areEqualArray(test_cell1_real3x3._internalSpan(), m_cell1_real3x3._internalSpan(), "CompareReal3x3");
  vc.areEqualArray(test2_cell1_real2._internalSpan(), m_cell1_real2._internalSpan(), "CompareReal2 Test2");

  vc.areEqualArray(test_cell1_array._internalSpan(), m_cell_array1._internalSpan(), "CompareRealArray1");
  {
    Span2<const double> v1(test_cell2_array.asArray());
    Span2<const double> v2(m_cell_array2.asArray());
    vc.areEqualArray(v1, v2, "CompareRealArray2");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestVariableFill()
{
  info() << "Execute Test VariableFill";

  ValueChecker vc(A_FUNCINFO);

  RunQueue queue = makeQueue(m_runner);
  RunQueue::ScopedAsync sca(&queue);

  const Real real_ref(1.9e-2);
  const Real3 real3_ref(2.3, 4.9e-2, -1.2);
  const Int32 nb_dim2 = 5;
  VariableCellReal test_cell1_real(VariableBuildInfo(mesh(), "TestCell1Real"));
  VariableCellArrayReal3 test_cell2_array_real3(VariableBuildInfo(mesh(), "TestCell2ArrayReal3"));
  test_cell2_array_real3.resize(nb_dim2);
  test_cell1_real.fill(real_ref, &queue);
  test_cell2_array_real3.fill(real3_ref, &queue);

  queue.barrier();

  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real r = test_cell1_real[cell];
    if (r != real_ref)
      ARCANE_FATAL("Bad value 1 cell={0} v={1} expected={2}", cell.uniqueId(), r, real_ref);
    for (Int32 z = 0; z < nb_dim2; ++z) {
      Real3 r3 = test_cell2_array_real3(cell, z);
      if (test_cell1_real[cell] != real_ref)
        ARCANE_FATAL("Bad value 2 cell={0} v={1} expected={2}", cell.uniqueId(), r3, real3_ref);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestGroupIndexTable()
{
  info() << "Execute GroupIndexTable";
  ItemGroup cells = allCells();
  SharedPtrT<GroupIndexTable> index_table = cells.localIdToIndex();
  GroupIndexTableView index_table_view = index_table->view();
  IItemFamily* cell_family = mesh()->cellFamily();
  const Int32 max_id = cell_family->maxLocalId();
  NumArray<Int32, MDDim1> ids(max_id);

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto out_ids = viewOut(command, ids);
    command << RUNCOMMAND_ENUMERATE (Cell, icell, allCells())
    {
      Int32 id1 = icell.localId();
      Int32 id2 = index_table_view[id1];
      out_ids[id1] = id2;
    };
  }

  ENUMERATE_ (Cell, icell, allCells()) {
    Int32 lid = icell.itemLocalId();
    Int32 x = index_table_view[lid];
    if (x != lid)
      ARCANE_FATAL("Bad id index={0} ref={1} v={2}", icell.index(), lid, x);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestMultiArray()
{
  info() << "Execute test TestMultiArray";
  ValueChecker vc(A_FUNCINFO);
  auto queue = makeQueue(m_runner);
  UniqueMultiArray2<Int32> values(MemoryUtils::getDefaultDataAllocator());
  const Int32 dim1_size = 5;
  NumArray<Int32, MDDim1> result_values(dim1_size);
  NumArray<Int32, MDDim1> expected_result_values(dim1_size);
  UniqueArray<Int32> sizes(dim1_size);
  for (Int32 i = 0; i < dim1_size; ++i)
    sizes[i] = i + 1;
  values.resize(sizes);
  MultiArray2SmallSpan<Int32> inout_values = values.span();
  for (Int32 i = 0; i < dim1_size; ++i) {
    const Int32 dim2_size = inout_values[i].size();
    Int32 total = 0;
    info() << "I=" << i << " dim2_size=" << dim2_size;
    for (Int32 j = 0; j < dim2_size; ++j) {
      Int32 v = i + j + 1;
      inout_values[i][j] = v;
      total += v;
    }
    expected_result_values[i] = total;
  }

  {
    auto command = makeCommand(queue);
    auto out_values_view = viewOut(command, result_values);

    command << RUNCOMMAND_LOOP1(iter, dim1_size)
    {
      auto [i] = iter();
      const Int32 dim2_size = inout_values[i].size();
      Int32 total = 0;
      for (Int32 j = 0; j < dim2_size; ++j)
        total += inout_values[i][j];
      out_values_view[i] = total;
    };
  }
  vc.areEqualArray(result_values.to1DSpan(), expected_result_values.to1DSpan(), "Result");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
