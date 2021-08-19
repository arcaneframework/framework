// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/accelerator/Views.h"
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

 private:

  void _setCellArrayValue(Integer seed);
  void _checkCellArrayValue(const String& message) const;
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

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
