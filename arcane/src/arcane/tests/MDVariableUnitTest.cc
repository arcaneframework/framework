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
#include "arcane/utils/NumVector.h"
#include "arcane/utils/NumMatrix.h"

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
  void _testVectorMDVariable();
  void _testMatrixMDVariable();
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
  _testVectorMDVariable();
  _testMatrixMDVariable();
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
  using MyVariableRef1 = MeshMDVariableRefT<Cell, Real, MDDim1>;
  using MyVariableRef2 = MeshMDVariableRefT<Cell, Real, MDDim2>;
  using MyVariableRef3 = MeshMDVariableRefT<Cell, Real, MDDim3>;

  {
    MyVariableRef2 my_var(VariableBuildInfo(mesh(), "TestCustomVar2D"));
    my_var.reshape({ 3, 4 });
    info() << "MyCustomVar=" << my_var.name();
    ENUMERATE_ (Cell, icell, allCells()) {
      my_var(icell, 1, 2) = 3.0;
      Real x = my_var(icell, 1, 2);
      if (x != 3.0)
        ARCANE_FATAL("Bad value (2)");
    }

    // Teste vue 3D d'une variable avec shape 2D
    MyVariableRef3 my_var3(VariableBuildInfo(mesh(), "TestCustomVar2D"));
    ENUMERATE_ (Cell, icell, allCells()) {
      Real x = my_var3(icell, 1, 2, 0);
      if (x != 3.0)
        ARCANE_FATAL("Bad value (3)");
    }
  }
  {
    MyVariableRef1 my_var1(VariableBuildInfo(mesh(), "TestCustomVar1D"));
    const Int32 size2 = 12;
    my_var1.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i)
        my_var1(icell, i) = static_cast<Real>(i + 1);
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real ref_value = static_cast<Real>(i + 1);
        Real r = my_var1(icell, i);
        if (r != ref_value)
          ARCANE_FATAL("Bad value (4)");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
_testVectorMDVariable()
{
  info() << "TEST VECTOR VARIABLE";
  using MyVariableRef1 = MeshVectorMDVariableRefT<Cell, Real, 3, MDDim1>;

  {
    MyVariableRef1 my_var1(VariableBuildInfo(mesh(), "TestCustomVectorVar1D"));

    const Int32 size2 = 12;
    my_var1.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN3 x(r0, r0 + 1.5, r0 + 2.3);
        my_var1(icell, i) = x;
      }
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN3 ref_value(r0, r0 + 1.5, r0 + 2.3);
        RealN3 r = my_var1(icell, i);
        if (r != ref_value)
          ARCANE_FATAL("Bad value (4)");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
_testMatrixMDVariable()
{
  info() << "TEST MATRIX VARIABLE";
  using MyVariableRef1 = MeshMatrixMDVariableRefT<Cell, Real, 2, 2, MDDim1>;

  {
    MyVariableRef1 my_var1(VariableBuildInfo(mesh(), "TestCustomMatrixVar1D"));

    const Int32 size2 = 7;
    my_var1.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN2x2 x({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
        my_var1(icell, i) = x;
      }
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN2x2 ref_value({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
        RealN2x2 r = my_var1(icell, i);
        if (r != ref_value)
          ARCANE_FATAL("Bad value (5)");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
