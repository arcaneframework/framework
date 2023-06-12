// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorItemInfoUnitTest.cc                              (C) 2000-2023 */
/*                                                                           */
/* Service de test unitaire des 'ItemGenericInfoListView'.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemGenericInfoListView.h"
#include "arcane/core/IMesh.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/VariableViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class AcceleratorItemInfoUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorItemInfoUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override {}
  void executeTest() override;

 public:

  void _executeTest1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorItemInfoUnitTest,
                                           IUnitTest, AcceleratorItemInfoUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorItemInfoUnitTest::
AcceleratorItemInfoUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorItemInfoUnitTest::
executeTest()
{
  _executeTest1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorItemInfoUnitTest::
_executeTest1()
{
  VariableCellInt64 var_unique_ids(VariableBuildInfo(mesh(), "TestUniqueIds"));
  VariableCellInt16 var_type_ids(VariableBuildInfo(mesh(), "TestTypeIds"));
  VariableCellInt32 var_owners(VariableBuildInfo(mesh(), "TestOwners"));

  VariableCellInt64 var_unique_ids2(VariableBuildInfo(mesh(), "TestUniqueIds2"));
  VariableCellInt16 var_type_ids2(VariableBuildInfo(mesh(), "TestTypeIds2"));
  VariableCellInt32 var_owners2(VariableBuildInfo(mesh(), "TestOwners2"));

  auto* queue = subDomain()->acceleratorMng()->defaultQueue();
  ItemGenericInfoListView cells_info(mesh()->cellFamily());
  {
    auto command = makeCommand(queue);

    auto out_unique_ids = viewOut(command, var_unique_ids);
    auto out_type_ids = viewOut(command, var_type_ids);
    auto out_owners = viewOut(command, var_owners);

    auto out_unique_ids2 = viewOut(command, var_unique_ids2);
    auto out_type_ids2 = viewOut(command, var_type_ids2);
    auto out_owners2 = viewOut(command, var_owners2);

    command << RUNCOMMAND_ENUMERATE (Cell, vi, allCells())
    {
      Int32 lid = vi.localId();
      out_unique_ids[vi] = cells_info.uniqueId(vi);
      out_type_ids[vi] = cells_info.typeId(vi);
      out_owners[vi] = cells_info.owner(vi);

      out_unique_ids2[vi] = cells_info.uniqueId(lid);
      out_type_ids2[vi] = cells_info.typeId(lid);
      out_owners2[vi] = cells_info.owner(lid);
    };
  }

  // Vérification
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;

    if (cell.uniqueId() != var_unique_ids[cell])
      ARCANE_FATAL("Bad 1 uniqueId() expected={0} current={1}", cell.uniqueId(), var_unique_ids[cell]);
    if (cell.itemTypeId() != var_type_ids[cell])
      ARCANE_FATAL("Bad 1 typeId() expected={0} current={1}", cell.itemTypeId(), var_type_ids[cell]);
    if (cell.owner() != var_owners[cell])
      ARCANE_FATAL("Bad 1 owner() expected={0} current={1}", cell.owner(), var_owners[cell]);

    if (cell.uniqueId() != var_unique_ids2[cell])
      ARCANE_FATAL("Bad 2 uniqueId() expected={0} current={1}", cell.uniqueId(), var_unique_ids[cell]);
    if (cell.itemTypeId() != var_type_ids2[cell])
      ARCANE_FATAL("Bad 2 typeId() expected={0} current={1}", cell.itemTypeId(), var_type_ids[cell]);
    if (cell.owner() != var_owners2[cell])
      ARCANE_FATAL("Bad 2 owner() expected={0} current={1}", cell.owner(), var_owners[cell]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
