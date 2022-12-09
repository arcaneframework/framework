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

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

template <typename ItemType, typename DataType, typename ExtentType>
class ItemVariableArrayAsMDRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType, typename MDDimType>
class ItemVariableArrayAsMDRefDynamicBaseT
: public VariableRef
{
 private:

  class CustomVariableRef
  : public MeshVariableArrayRefT<ItemType, DataType>
  {
   public:

    using BaseClass = MeshVariableArrayRefT<ItemType, DataType>;
    using VariableType = typename BaseClass::PrivatePartType;
    using ValueDataType = typename VariableType::ValueDataType;

   public:

    explicit CustomVariableRef(const VariableBuildInfo& vbi)
    : BaseClass(vbi)
    {
    }

   public:

    ValueDataType* trueData() { return this->m_private_part->trueData(); }
  };

 public:

  using UnderlyingVariableType = MeshVariableArrayRefT<ItemType, DataType>;
  using MDSpanType = MDSpan<DataType, MDDimType, RightLayout>;
  using ItemLocalIdType = typename ItemType::LocalIdType;

 public:

  explicit ItemVariableArrayAsMDRefDynamicBaseT(const VariableBuildInfo& b)
  : VariableRef()
  , m_underlying_var(b)
  {
    _internalAssignVariable(m_underlying_var);
  }

  UnderlyingVariableType& underlyingVariable() { return m_underlying_var; }

 protected:

  void updateFromInternal() override
  {
    // ATTENTION à ne pas utiliser underlying_var directement car la vue
    // associée sur les entités n'est pas forcément remise à jour.
    IData* data = this->m_underlying_var.variable()->data();
    ArrayShape shape = data->shape();
    //std::cout << "SHAPE=" << shape.dimensions() << "\n";
    auto* true_data = m_underlying_var.trueData();
    auto array_view = true_data->view();
    Int32 dim0_size = array_view.dim1Size();
    m_mdspan = MDSpanType(array_view.data(), { dim0_size, shape.dimension(0), shape.dimension(1) });
  }

 protected:

  CustomVariableRef m_underlying_var;
  MDSpanType m_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType>
class ItemVariableArrayAsMDRefT<ItemType, DataType, MDDim2>
: public ItemVariableArrayAsMDRefDynamicBaseT<ItemType, DataType, MDDim3>
{
 public:

  using BaseClass = ItemVariableArrayAsMDRefDynamicBaseT<ItemType, DataType, MDDim3>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using MDSpanType = typename BaseClass::MDSpanType;

 public:

  explicit ItemVariableArrayAsMDRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2)
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }
  const DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2) const
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }

  void reshape(Int32 dim1, Int32 dim2)
  {
    ArrayShape shape;
    shape.setNbDimension(2);
    shape.setDimension(0, dim1);
    shape.setDimension(1, dim2);
    //this->m_underlying_var.variable()->data()->setShape(shape);
    this->m_underlying_var.resizeAndReshape(shape);
  }
};

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
  using MyVariableRef2 = ItemVariableArrayAsMDRefT<Cell, Real, MDDim2>;

  MyVariableRef2 my_var(VariableBuildInfo(mesh(), "TestCustomVar"));
  my_var.reshape(3, 4);
  info() << "MyCustomVar=" << my_var.name();
  ENUMERATE_ (Cell, icell, allCells()) {
    my_var(icell, 1, 2) = 3.0;
    Real x = my_var(icell, 1, 2);
    if (x != 3.0)
      ARCANE_FATAL("Bad value");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
