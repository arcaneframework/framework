// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDVariableUnitTest.cc                                       (C) 2000-2023 */
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

  void samples1();
  void samples2();
  void samples3();

 private:

  void _testCustomVariable();
  void _testVectorMDVariable();
  void _testMatrixMDVariable();

  template <typename VarType> void
  _setAndTestShape(VarType& var_ref,
                   std::array<Int32, VarType::nb_dynamic> reshape_value,
                   std::array<Int32, VarType::FullExtentsType::rank() - 1> expected_shape);
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
  samples1();
  samples2();
  samples3();
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
      Real rr0 = m_vector_var0d(icell)(0);
      Real rr1 = m_vector_var0d(icell)(1);
      Real rr2 = m_vector_var0d(icell)(2);
      if (rr0 != ref_value(0))
        ARCANE_FATAL("Bad valueX (1)");
      if (rr1 != ref_value(1))
        ARCANE_FATAL("Bad valueY (1)");
      if (rr2 != ref_value(2))
        ARCANE_FATAL("Bad valueZ (1)");
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

      Real rr00 = m_matrix_var0d(icell)(0, 0);
      Real rr01 = m_matrix_var0d(icell)(0, 1);

      Real rr10 = m_matrix_var0d(icell)(1, 0);
      Real rr11 = m_matrix_var0d(icell)(1, 1);

      if (rr00 != ref_value(0, 0))
        ARCANE_FATAL("Bad valueXX (1)");
      if (rr01 != ref_value(0, 1))
        ARCANE_FATAL("Bad valueXY (1)");
      if (rr10 != ref_value(1, 0))
        ARCANE_FATAL("Bad valueYX (1)");
      if (rr11 != ref_value(1, 1))
        ARCANE_FATAL("Bad valueYY (1)");

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

  _setAndTestShape(m_scalar_var0d_real, {}, {});
  _setAndTestShape(m_scalar_var1d_real, { 19 }, { 19 });
  _setAndTestShape(m_scalar_var2d_real, { 7, 13 }, { 7, 13 });
  _setAndTestShape(m_scalar_var2d_as_3d_real, { 5, 11, 14 }, { 5, 11, 14 });
  _setAndTestShape(m_vector_var0d_real2, {}, { 2 });
  _setAndTestShape(m_vector_var1d_real3, { 9 }, { 9, 3 });
  _setAndTestShape(m_vector_var2d_real4, { 23, 12 }, { 23, 12, 4 });
  _setAndTestShape(m_matrix_var0d_real2x2, {}, { 2, 2 });
  _setAndTestShape(m_matrix_var1d_real2x6, { 7 }, { 7, 2, 6 });
  _setAndTestShape(m_matrix_var0d_real3x2, {}, { 3, 2 });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename VarType> void MDVariableUnitTest::
_setAndTestShape(VarType& var_ref, std::array<Int32, VarType::nb_dynamic> reshape_value,
                 std::array<Int32, VarType::FullExtentsType::rank() - 1> expected_shape_array)
{
  var_ref.reshape(reshape_value);
  ArrayShape full_shape(var_ref.fullShape());
  info() << "Shape name=" << var_ref.name() << " shape=" << full_shape;
  Span<const Int32> shape_span(expected_shape_array);
  ArrayShape expected_shape(shape_span);
  if (full_shape != expected_shape)
    ARCANE_FATAL("Bad shape for variable '{0}' shape={1} expected={2}",
                 var_ref.name(), full_shape, expected_shape);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
samples1()
{
  //![SampleMDVariableScalar]
  // Déclare une variable 2D aux mailles de type Real
  //
  // La déclaration équivalente dans le fichier AXL est:
  // <variable field-name="cell_var_2d" name="Var1"
  //           data-type="real" item-kind="cell" shape-dim="2"
  // />
  Arcane::VariableBuildInfo vbi(VariableBuildInfo(mesh(), "Var1"));
  Arcane::MeshMDVariableRefT<Cell, Real, MDDim2> cell_var_2d(vbi);

  // Positionne les deux dimensions à 3x4.
  // Chaque maille aura 3x4 = 12 valeurs
  cell_var_2d.reshape({ 3, 4 });

  ENUMERATE_ (Cell, icell, allCells()) {
    // Positionne la valeur pour l'indice (2,1)
    cell_var_2d(icell, 2, 1) = 2.3;

    // Affiche la valeur pour l'indice (2,1)
    info() << cell_var_2d(icell, 2, 1);
  }
  //![SampleMDVariableScalar]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
samples2()
{
  //![SampleMDVariableVector]
  // Déclare une variable 2D aux mailles d'un vecteur de 7 Real
  //
  // La déclaration équivalente dans le fichier AXL est:
  // <variable field-name="cell_var_2d" name="Var1"
  //           data-type="real" item-kind="cell" shape-dim="2"
  //           extent0="7"
  // />
  Arcane::VariableBuildInfo vbi(VariableBuildInfo(mesh(), "Var1"));
  Arcane::MeshVectorMDVariableRefT<Cell, Real, 7, MDDim2> cell_var_2d(vbi);

  // Positionne les deux dimensions à 2x3.
  // Chaque maille aura 2x3 = 6 valeurs de 7 réels
  cell_var_2d.reshape({ 3, 4 });

  ENUMERATE_ (Cell, icell, allCells()) {
    Arcane::NumVector<Real, 7> v({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    // Positionne la valeur pour l'indice (2,1)
    cell_var_2d(icell, 2, 1) = v;

    // Affiche la valeur du 6-ème élément du vecteur pour l'indice (2,1)
    info() << cell_var_2d(icell, 2, 1)(5);
  }
  //![SampleMDVariableVector]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MDVariableUnitTest::
samples3()
{
  //![SampleMDVariableMatrix]
  // Déclare une variable 1D aux mailles d'une matrix 2x5 de Real
  //
  // La déclaration équivalente dans le fichier AXL est:
  // <variable field-name="cell_var_2d" name="Var1"
  //           data-type="real" item-kind="cell" shape-dim="2"
  //           extent0="2" extent1="5"
  // />
  Arcane::VariableBuildInfo vbi(VariableBuildInfo(mesh(), "Var1"));
  Arcane::MeshMatrixMDVariableRefT<Cell, Real, 2, 5, MDDim1> cell_var_2d(vbi);

  // Positionne la dimension à 9.
  // Chaque maille aura 9 valeurs de 2x5 = 10 réels
  cell_var_2d.reshape({ 9 });

  ENUMERATE_ (Cell, icell, allCells()) {
    Arcane::NumMatrix<Real, 2, 5> v({ 1.1, 2.2, 3.3, 4.4, 5.5 },
                                    { 1.0, 2.0, 3.0, 4.0, 5.0 });

    // Positionne la valeur pour l'indice (2,1)
    cell_var_2d(icell, 7) = v;

    // Affiche la valeur du (1,4) de la matrice pour l'indice (6)
    info() << cell_var_2d(icell, 6)(1, 4);
  }
  //![SampleMDVariableMatrix]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
