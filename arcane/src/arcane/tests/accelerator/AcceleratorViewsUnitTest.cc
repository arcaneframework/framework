// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorViewsUnitTest.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Service de test des vues pour les accelerateurs.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/VariableViews.h"
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
  VariableCellArrayReal m_cell_array1;
  VariableCellArrayReal m_cell_array2;
  VariableCellReal2 m_cell1_real2;
  VariableCellReal3 m_cell1_real3;
  VariableCellReal2x2 m_cell1_real2x2;
  VariableCellReal3x3 m_cell1_real3x3;

 private:

  void _setCellArrayValue(Integer seed);
  void _checkCellArrayValue(const String& message) const;

 public:

  void _executeTest1();
  void _executeTest2();
  void _executeTestReal2x2();
  void _executeTestReal3x3();
  void _executeTest2Real3x3();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorViewsUnitTest,IUnitTest,
                                           AcceleratorViewsUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorViewsUnitTest::
AcceleratorViewsUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_cell_array1(VariableBuildInfo(sb.mesh(),"CellArray1"))
, m_cell_array2(VariableBuildInfo(sb.mesh(),"CellArray2"))
, m_cell1_real2(VariableBuildInfo(sb.mesh(),"Cell1Real2"))
, m_cell1_real3(VariableBuildInfo(sb.mesh(),"Cell1Real3"))
, m_cell1_real2x2(VariableBuildInfo(sb.mesh(),"Cell1Real2x2"))
, m_cell1_real3x3(VariableBuildInfo(sb.mesh(),"Cell1Real3x3"))
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
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner,traceMng(),acc_info);

  m_cell_array1.resize(12);
  m_cell_array2.resize(12);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_setCellArrayValue(Integer seed)
{
  Integer n = m_cell_array1.arraySize();
  ENUMERATE_CELL(icell,allCells()){
    Int32 lid = icell.itemLocalId();
    for (Integer i=0; i<n; ++i ){
      m_cell_array1[icell][i] = Real((i+1) + lid + seed);
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
  ENUMERATE_CELL(icell,allCells()){
    vc.areEqual(m_cell_array1[icell],m_cell_array2[icell],message);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
executeTest()
{
  _executeTest1();
  _executeTest2();
  _executeTestReal2x2();
  _executeTestReal3x3();
  _executeTest2Real3x3();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest1()
{
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);

  Integer dim2_size = m_cell_array1.arraySize();

  {
    int seed = 37;
    _setCellArrayValue(seed);

    auto in_cell_array1 = ax::viewIn(command,m_cell_array1);
    auto out_cell_array2 = ax::viewOut(command,m_cell_array2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      out_cell_array2[vi].copy(in_cell_array1[vi]);
    };

    _checkCellArrayValue("View1");
  }

  {
    int seed = 23;
    _setCellArrayValue(seed);

    auto in_cell_array1 = ax::viewIn(command,m_cell_array1);
    auto out_cell_array2 = ax::viewOut(command,m_cell_array2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      for (Integer i=0; i<dim2_size; ++i )
        out_cell_array2[vi][i] = in_cell_array1[vi][i];
    };
    _checkCellArrayValue("View2");
  }

  {
    int seed = 53;
    _setCellArrayValue(seed);

    auto in_cell_array1 = ax::viewInOut(command,m_cell_array1);
    auto out_cell_array2 = ax::viewOut(command,m_cell_array2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      out_cell_array2[vi].copy(in_cell_array1[vi]);
    };

    _checkCellArrayValue("View3");
  }

  {
    int seed = 93;
    _setCellArrayValue(seed);

    auto in_cell_array1 = ax::viewIn(command,m_cell_array1);
    auto out_cell_array2 = ax::viewInOut(command,m_cell_array2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      out_cell_array2[vi].copy(in_cell_array1[vi]);
    };

    _checkCellArrayValue("View4");
  }

  {
    int seed = 43;
    _setCellArrayValue(seed);

    auto inout_cell_array1 = ax::viewInOut(command,m_cell_array1);
    auto out_cell_array2 = ax::viewInOut(command,m_cell_array2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
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
  ValueChecker vc(A_FUNCINFO);
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2 = ax::viewInOut(command,m_cell1_real2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2 ref_v;
      ref_v[0] = 2.0+v;
      ref_v[1] = 3.0+v;
      inout_cell1_real2[vi].setX(ref_v[0]);
      inout_cell1_real2[vi].setY(ref_v[1]);
    };
  }
  {
    ENUMERATE_(Cell,vi,allCells()){
      Real v = static_cast<Real>(vi.itemLocalId());
      Real rx = 2.0+v;
      Real ry = 3.0+v;
      vc.areEqual(rx,m_cell1_real2[vi].x,"CheckX");
      vc.areEqual(ry,m_cell1_real2[vi].y,"CheckY");
    }
  }

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3 = ax::viewInOut(command,m_cell1_real3);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3 ref_v;
      ref_v[0] = 2.0+v;
      ref_v[1] = 3.0+v;
      ref_v[2] = 4.0+v;
      inout_cell1_real3[vi].setX(ref_v[0]);
      inout_cell1_real3[vi].setY(ref_v[1]);
      inout_cell1_real3[vi].setZ(ref_v[2]);
    };
  }
  {
    ENUMERATE_(Cell,vi,allCells()){
      Real v = static_cast<Real>(vi.itemLocalId());
      Real rx = 2.0+v;
      Real ry = 3.0+v;
      Real rz = 4.0+v;
      vc.areEqual(rx,m_cell1_real3[vi].x,"CheckX");
      vc.areEqual(ry,m_cell1_real3[vi].y,"CheckY");
      vc.areEqual(rz,m_cell1_real3[vi].z,"CheckZ");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestReal2x2()
{
  info() << "Execute Test Real2x2";
  ValueChecker vc(A_FUNCINFO);
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real2x2 = ax::viewInOut(command,m_cell1_real2x2);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real2x2 ref_v;
      ref_v[0][0] = 2.0+v;
      ref_v[1][0] = 3.0+v;
      ref_v[0][1] = 4.0+v;
      ref_v[1][1] = 5.0+v;
      inout_cell1_real2x2[vi].setXX(ref_v[0][0]);
      inout_cell1_real2x2[vi].setYX(ref_v[1][0]);
      inout_cell1_real2x2[vi].setXY(ref_v[0][1]);
      inout_cell1_real2x2[vi].setYY(ref_v[1][1]);
    };
  }
  {
    ENUMERATE_(Cell,vi,allCells()){
      Real v = static_cast<Real>(vi.itemLocalId());
      Real rxx = 2.0+v;
      Real ryx = 3.0+v;
      Real rxy = 4.0+v;
      Real ryy = 5.0+v;
      vc.areEqual(rxx,m_cell1_real2x2[vi].x.x,"Real2x2CheckXX");
      vc.areEqual(rxy,m_cell1_real2x2[vi].x.y,"Real2x2CheckXY");
      vc.areEqual(ryx,m_cell1_real2x2[vi].y.x,"Real2x2CheckYX");
      vc.areEqual(ryy,m_cell1_real2x2[vi].y.y,"Real2x2CheckYY");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTestReal3x3()
{
  info() << "Execute Test Real3x3";
  ValueChecker vc(A_FUNCINFO);
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3x3 = ax::viewInOut(command,m_cell1_real3x3);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3x3 ref_v;
      ref_v[0][0] = 2.0+v;
      ref_v[1][0] = 3.0+v;
      ref_v[2][0] = 4.0+v;

      ref_v[0][1] = 5.0+v;
      ref_v[1][1] = 6.0+v;
      ref_v[2][1] = 7.0+v;

      ref_v[0][2] = 8.0+v;
      ref_v[1][2] = 9.0+v;
      ref_v[2][2] = 10.0+v;

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
  {
    ENUMERATE_(Cell,vi,allCells()){
      Real v = static_cast<Real>(vi.itemLocalId());
      Real rxx = 2.0+v;
      Real ryx = 3.0+v;
      Real rzx = 4.0+v;
      Real rxy = 5.0+v;
      Real ryy = 6.0+v;
      Real rzy = 7.0+v;
      Real rxz = 8.0+v;
      Real ryz = 9.0+v;
      Real rzz = 10.0+v;
      vc.areEqual(rxx,m_cell1_real3x3[vi].x.x,"Real3x3CheckXX");
      vc.areEqual(ryx,m_cell1_real3x3[vi].y.x,"Real3x3CheckYX");
      vc.areEqual(rzx,m_cell1_real3x3[vi].z.x,"Real3x3CheckZX");

      vc.areEqual(rxy,m_cell1_real3x3[vi].x.y,"Real3x3CheckXY");
      vc.areEqual(ryy,m_cell1_real3x3[vi].y.y,"Real3x3CheckYY");
      vc.areEqual(rzy,m_cell1_real3x3[vi].z.y,"Real3x3CheckZY");

      vc.areEqual(rxz,m_cell1_real3x3[vi].x.z,"Real3x3CheckXZ");
      vc.areEqual(ryz,m_cell1_real3x3[vi].y.z,"Real3x3CheckYZ");
      vc.areEqual(rzz,m_cell1_real3x3[vi].z.z,"Real3x3CheckZZ");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorViewsUnitTest::
_executeTest2Real3x3()
{
  info() << "Execute Test2 Real3x3";
  ValueChecker vc(A_FUNCINFO);
  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto inout_cell1_real3x3 = ax::viewInOut(command,m_cell1_real3x3);

    command << RUNCOMMAND_ENUMERATE(Cell,vi,allCells())
    {
      Real v = static_cast<Real>(vi.localId());
      Real3x3 ref_v;
      ref_v[0][0] = 5.0+v;
      ref_v[1][0] = 6.0+v;
      ref_v[2][0] = 7.0+v;

      ref_v[0][1] = 8.0+v;
      ref_v[1][1] = 9.0+v;
      ref_v[2][1] = 10.0+v;

      ref_v[0][2] = 11.0+v;
      ref_v[1][2] = 12.0+v;
      ref_v[2][2] = 13.0+v;

      inout_cell1_real3x3[vi].setX(ref_v[0]);
      inout_cell1_real3x3[vi].setY(ref_v[1]);
      inout_cell1_real3x3[vi].setZ(ref_v[2]);
    };
  }
  {
    ENUMERATE_(Cell,vi,allCells()){
      Real v = static_cast<Real>(vi.itemLocalId());
      Real rxx = 5.0+v;
      Real ryx = 6.0+v;
      Real rzx = 7.0+v;
      Real rxy = 8.0+v;
      Real ryy = 9.0+v;
      Real rzy = 10.0+v;
      Real rxz = 11.0+v;
      Real ryz = 12.0+v;
      Real rzz = 13.0+v;
      vc.areEqual(rxx,m_cell1_real3x3[vi].x.x,"Real3x3CheckXX");
      vc.areEqual(ryx,m_cell1_real3x3[vi].y.x,"Real3x3CheckYX");
      vc.areEqual(rzx,m_cell1_real3x3[vi].z.x,"Real3x3CheckZX");

      vc.areEqual(rxy,m_cell1_real3x3[vi].x.y,"Real3x3CheckXY");
      vc.areEqual(ryy,m_cell1_real3x3[vi].y.y,"Real3x3CheckYY");
      vc.areEqual(rzy,m_cell1_real3x3[vi].z.y,"Real3x3CheckZY");

      vc.areEqual(rxz,m_cell1_real3x3[vi].x.z,"Real3x3CheckXZ");
      vc.areEqual(ryz,m_cell1_real3x3[vi].y.z,"Real3x3CheckYZ");
      vc.areEqual(rzz,m_cell1_real3x3[vi].z.z,"Real3x3CheckZZ");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
