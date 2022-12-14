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
#include "arcane/tests/MDVariableUnitTest_axl.h"

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
: public ArcaneMDVariableUnitTestObject
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
: ArcaneMDVariableUnitTestObject(sbi)
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

  {
    // Teste variable 2D
    m_scalar_var2d.reshape({ 3, 4 });
    info() << "MyCustomVar=" << m_scalar_var2d.name();
    ENUMERATE_ (Cell, icell, allCells()) {
      m_scalar_var2d(icell, 1, 2) = 3.0;
      Real x = m_scalar_var2d(icell, 1, 2);
      if (x != 3.0)
        ARCANE_FATAL("Bad value (2)");
    }

    // Teste vue 3D d'une variable avec shape 2D
    ENUMERATE_ (Cell, icell, allCells()) {
      Real x = m_scalar_var2d_as_3d(icell, 1, 2, 0);
      if (x != 3.0)
        ARCANE_FATAL("Bad value (3)");
    }
  }
  {
    // Teste variable 1D
    const Int32 size2 = 12;
    m_scalar_var1d.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i)
        m_scalar_var1d(icell, i) = static_cast<Real>(i + 1);
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real ref_value = static_cast<Real>(i + 1);
        Real r = m_scalar_var1d(icell, i);
        if (r != ref_value)
          ARCANE_FATAL("Bad value (4)");
      }
    }
  }
  {
    // Teste variable 0D
    m_scalar_var1d.reshape({});
    ENUMERATE_ (Cell, icell, allCells()) {
      m_scalar_var0d(icell) = static_cast<Real>(icell.itemLocalId() + 1);
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      Real ref_value = static_cast<Real>(icell.itemLocalId() + 1);
      Real r = m_scalar_var0d(icell);
      if (r != ref_value)
        ARCANE_FATAL("Bad value (4)");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
_testVectorMDVariable()
{
  info() << "TEST VECTOR VARIABLE";

  // Variable 0D
  {
    m_vector_var0d.reshape({});
    ENUMERATE_ (Cell, icell, allCells()) {
      Real r0 = static_cast<Real>(icell.itemLocalId() + 1);
      RealN3 x(r0, r0 + 1.5, r0 + 2.3);
      m_vector_var0d(icell) = x;
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      Real r0 = static_cast<Real>(icell.itemLocalId() + 1);
      RealN3 ref_value(r0, r0 + 1.5, r0 + 2.3);
      RealN3 r = m_vector_var0d(icell);
      if (r != ref_value)
        ARCANE_FATAL("Bad value (1)");
    }
  }

  // Variable 1D
  {
    const Int32 size2 = 12;
    m_vector_var1d.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN3 x(r0, r0 + 1.5, r0 + 2.3);
        m_vector_var1d(icell, i) = x;
      }
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN3 ref_value(r0, r0 + 1.5, r0 + 2.3);
        RealN3 r = m_vector_var1d(icell, i);
        if (r != ref_value)
          ARCANE_FATAL("Bad value (2)");
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

  // Variable 0D
  {
    m_matrix_var0d.reshape({});
    ENUMERATE_ (Cell, icell, allCells()) {
      Real r0 = static_cast<Real>(icell.itemLocalId() + 1);
      RealN2x2 x({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
      m_matrix_var0d(icell) = x;
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      Real r0 = static_cast<Real>(icell.itemLocalId() + 1);
      RealN2x2 ref_value({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
      RealN2x2 r = m_matrix_var0d(icell);
      if (r != ref_value)
        ARCANE_FATAL("Bad value (5)");
    }
  }

  // Variable 1D
  {
    const Int32 size2 = 7;
    m_matrix_var1d.reshape({ size2 });
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN2x2 x({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
        m_matrix_var1d(icell, i) = x;
      }
    }
    ENUMERATE_ (Cell, icell, allCells()) {
      for (Int32 i = 0; i < size2; ++i) {
        Real r0 = static_cast<Real>(i + 1);
        RealN2x2 ref_value({ r0, r0 + 1.5 }, { r0 + 2.3, r0 - 4.3 });
        RealN2x2 r = m_matrix_var1d(icell, i);
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
