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

template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefBaseT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType>
class CustomVariableRef
: public MeshVariableArrayRefT<ItemType, DataType>
{
  template <typename _ItemType, typename _DataType, typename _Extents>
  friend class MeshMDVariableRefBaseT;

 public:

  using BaseClass = MeshVariableArrayRefT<ItemType, DataType>;
  using VariableType = typename BaseClass::PrivatePartType;
  using ValueDataType = typename VariableType::ValueDataType;

 private:

  explicit CustomVariableRef(const VariableBuildInfo& vbi)
  : BaseClass(vbi)
  {
  }

 private:

  ValueDataType* trueData() { return this->m_private_part->trueData(); }

  void fillShape(ArrayShape& shape_with_item)
  {
    this->m_private_part->fillShape(shape_with_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefBaseT
: public MeshVariableRef
{
 public:

  using UnderlyingVariableType = MeshVariableArrayRefT<ItemType, DataType>;
  using MDSpanType = MDSpan<DataType, Extents, RightLayout>;
  using ItemLocalIdType = typename ItemType::LocalIdType;

 public:

  explicit MeshMDVariableRefBaseT(const VariableBuildInfo& b)
  : MeshVariableRef()
  , m_underlying_var(b)
  {
    _internalInit(m_underlying_var.variable());
  }

  UnderlyingVariableType& underlyingVariable() { return m_underlying_var; }

 protected:

  void updateFromInternal() override
  {
    const Int32 nb_rank = Extents::rank();
    ArrayShape shape_with_item;
    shape_with_item.setNbDimension(nb_rank);
    m_underlying_var.fillShape(shape_with_item);

    ArrayExtents<Extents> new_extents = ArrayExtentsBase<Extents>::fromSpan(shape_with_item.dimensions());
    m_mdspan = MDSpanType(m_underlying_var.trueData()->view().data(), new_extents);
  }

 protected:

  CustomVariableRef<ItemType, DataType> m_underlying_var;
  MDSpanType m_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType, typename Extents::AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = typename Extents::AddedFirstExtentsType<DynExtent>;

 public:

  using BaseClass = MeshMDVariableRefBaseT<ItemType, DataType, AddedFirstExtentsType>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using MDSpanType = typename BaseClass::MDSpanType;

 public:

  explicit MeshMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2)
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i1, Int32 i2) const
  {
    return this->m_mdspan(id.localId(), i1, i2);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i, Int32 j, Int32 k)
  {
    return this->m_mdspan(id.localId(), i, j, k);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i, Int32 j, Int32 k) const
  {
    return this->m_mdspan(id.localId(), i, j, k);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::nb_dynamic == 2, void>>
  void reshape(Int32 dim1, Int32 dim2)
  {
    ArrayShape shape;
    shape.setNbDimension(2);
    shape.setDimension(0, dim1);
    shape.setDimension(1, dim2);
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
  using MyVariableRef2 = MeshMDVariableRefT<Cell, Real, MDDim2>;
  using MyVariableRef3 = MeshMDVariableRefT<Cell, Real, MDDim3>;

  MyVariableRef2 my_var(VariableBuildInfo(mesh(), "TestCustomVar"));
  my_var.reshape(3, 4);
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
