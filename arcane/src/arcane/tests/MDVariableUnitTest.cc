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

template <typename ItemType, typename DataType, typename Extents>
class ItemVariableArrayAsMDRefDynamicBaseT
: public MeshVariableRef
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
  using MDSpanType = MDSpan<DataType, Extents, RightLayout>;
  using ItemLocalIdType = typename ItemType::LocalIdType;

 public:

  explicit ItemVariableArrayAsMDRefDynamicBaseT(const VariableBuildInfo& b)
  : MeshVariableRef()
  , m_underlying_var(b)
  {
    _internalInit(m_underlying_var.variable());
  }

  UnderlyingVariableType& underlyingVariable() { return m_underlying_var; }

 protected:
 public:

  void updateFromInternal() override
  {
    // ATTENTION à ne pas utiliser underlying_var directement car la vue
    // associée sur les entités n'est pas forcément remise à jour.
    IData* data = this->m_underlying_var.variable()->data();
    ArrayShape shape = data->shape();
    const Int32 nb_rank = Extents::rank();
    //std::cout << "SHAPE=" << shape.dimensions() << " internal_rank=" << nb_rank << "\n";
    auto* true_data = m_underlying_var.trueData();
    auto array_view = true_data->view();
    Int32 dim0_size = array_view.dim1Size();

    ArrayShape shape_with_item;
    shape_with_item.setNbDimension(nb_rank);
    shape_with_item.setDimension(0, dim0_size);
    Int32 nb_orig_shape = shape.nbDimension();
    for (Int32 i = 0; i < nb_orig_shape; ++i) {
      shape_with_item.setDimension(i + 1, shape.dimension(i));
    }
    // Si la forme est plus petite que notre rang, remplit les dimensions
    // supplémentaires par la valeur 1.
    for (Int32 i = (nb_orig_shape + 1); i < nb_rank; ++i) {
      shape_with_item.setDimension(i, 1);
    }

    //new_extents = ArrayExtentsBase<Extents>::fromSpan(shape.dimensions());
    ArrayExtents<Extents> new_extents = ArrayExtentsBase<Extents>::fromSpan(shape_with_item.dimensions());
    //m_mdspan = impl::_buildSpan<Rank, MDSpanType>(array_view.data(), dim0_size, shape);
    // MDSpanType(array_view.data(), { dim0_size, shape.dimension(0), shape.dimension(1) });
    m_mdspan = MDSpanType(array_view.data(), new_extents);
  }

 protected:

  CustomVariableRef m_underlying_var;
  MDSpanType m_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType, typename Extents>
class ItemVariableArrayAsMDRefT
: public ItemVariableArrayAsMDRefDynamicBaseT<ItemType, DataType, typename Extents::AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = typename Extents::AddedFirstExtentsType<DynExtent>;

 public:

  using BaseClass = ItemVariableArrayAsMDRefDynamicBaseT<ItemType, DataType, AddedFirstExtentsType>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using MDSpanType = typename BaseClass::MDSpanType;

 public:

  explicit ItemVariableArrayAsMDRefT(const VariableBuildInfo& b)
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
  using MyVariableRef2 = ItemVariableArrayAsMDRefT<Cell, Real, MDDim2>;
  using MyVariableRef3 = ItemVariableArrayAsMDRefT<Cell, Real, MDDim3>;

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
