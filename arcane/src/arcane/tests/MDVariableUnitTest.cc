// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDVariableUnitTest.cc                                       (C) 2000-2022 */
/*                                                                           */
/* Service de test des variables multi-dimension.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/Event.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MDSpan.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/IVariableMng.h"
#include "arcane/VariableTypes.h"
#include "arcane/ServiceFactory.h"
#include "arcane/MeshMDVariableRef.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des variables
 */
class MDVariableUnitTest
: public BasicUnitTest
{
 public:

  explicit MDVariableUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  void _testCustomVariable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MDVariableUnitTest,
                        Arcane::ServiceProperty("MDVariableUnitTest", Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(Arcane::IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MDVariableUnitTest::
MDVariableUnitTest(const ServiceBuildInfo& sbi)
: BasicUnitTest(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
executeTest()
{
  _testCustomVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
_testCustomVariable()
{
  info() << "TEST CUSTOM VARIABLE";
  using MyVariableRef2 = MeshMDVariableRefT<Cell, Real, MDDim2>;
  using MyVariableRef3 = MeshMDVariableRefT<Cell, Real, MDDim3>;

  MyVariableRef2 my_var(VariableBuildInfo(mesh(), "TestCustomVar"));
  my_var.reshape({ 3, 4 });
  info() << "MyCustomVar=" << my_var.name();
  ENUMERATE_ (Cell, icell, allCells()) {
    my_var(icell, 1, 2) = 3.0;
    Real x = my_var(icell, 1, 2);
    if (x != 3.0)
      ARCANE_FATAL("Bad value (2)");
  }
  MyVariableRef3 my_var3(VariableBuildInfo(mesh(), "TestCustomVar"));
  ENUMERATE_ (Cell, icell, allCells()) {
    Real x = my_var3(icell, 1, 2, 0);
    if (x != 3.0)
      ARCANE_FATAL("Bad value (3)");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
