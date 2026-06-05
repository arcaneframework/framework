// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableViews.h                                     (C) 2000-2026 */
/*                                                                           */
/* Management of views on material variables for accelerators.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
#define ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"
#include "arcane/core/materials/MeshEnvironmentVariableRef.h"
#include "arcane/core/materials/MatItem.h"

#include "arcane/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for views on material variables.
 */
class ARCANE_ACCELERATOR_EXPORT MatVariableViewBase
{
 public:

  // Currently does not use parameters yet
  MatVariableViewBase(const ViewBuildInfo&, IMeshMaterialVariable*);

  // Currently does not use parameters yet
  MatVariableViewBase() = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view on a scalar mesh variable.
 */
template <typename ItemType, typename DataType>
class MatItemVariableScalarInViewT
: public MatVariableViewBase
{
  // TODO: Should SIMD handling be added like in ItemVariableScalarInViewT?

 private:

  using ItemIndexType = typename ItemTraitsT<ItemType>::LocalIdType;

 public:

  MatItemVariableScalarInViewT(const ViewBuildInfo& vbi, IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(vbi, var)
  , m_value(v)
  {}
  MatItemVariableScalarInViewT() = default;

  //! Access operator for entity \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  //! Access operator for entity \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value[0][pmvi.valueIndex()];
  }

  //! Override to access the global value from the cell id
  ARCCORE_HOST_DEVICE const DataType& operator[](ItemIndexType item) const
  {
    return this->m_value[0][item.localId()];
  }

  //! Access operator for entity \a item
  ARCCORE_HOST_DEVICE const DataType& value(ComponentItemLocalId mvi) const
  {
    return this->m_value[mvi.localId().arrayIndex()][mvi.localId().valueIndex()];
  }

  ARCCORE_HOST_DEVICE const DataType& value0(PureMatVarIndex idx) const
  {
    return this->m_value[0][idx.valueIndex()];
  }

 private:

  ArrayView<DataType>* m_value = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Write view on a scalar mesh variable.
 */
template <typename ItemType, typename Accessor>
class MatItemVariableScalarOutViewT
: public MatVariableViewBase
{
 private:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnType = DataType&;
  using ItemIndexType = typename ItemTraitsT<ItemType>::LocalIdType;

  // TODO: Should ARCANE_CHECK_AT(mvi.arrayIndex(), m_value.size()); be added?
  // The check on the other dimension will still be missing.
  // TODO: Should SIMD type handling be added like in ItemVariableScalarOutViewT?

 public:

  MatItemVariableScalarOutViewT(const ViewBuildInfo& vbi, IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(vbi, var)
  , m_value(v)
  {}
  MatItemVariableScalarOutViewT() = default;

  //! Access operator for entity \a item
  ARCCORE_HOST_DEVICE Accessor operator[](ComponentItemLocalId lid) const
  {
    return Accessor(this->m_value[lid.localId().arrayIndex()].data() + lid.localId().valueIndex());
  }

  ARCCORE_HOST_DEVICE Accessor operator[](PureMatVarIndex pmvi) const
  {
    return Accessor(this->m_value[0][pmvi.valueIndex()]);
  }

  //! Override to access the global value from the cell id
  ARCCORE_HOST_DEVICE Accessor operator[](ItemIndexType item) const
  {
    ARCANE_CHECK_AT(item.localId(), this->m_value[0].size());
    return Accessor(this->m_value[0].data() + item.localId());
  }

  //! Access operator for entity \a item
  ARCCORE_HOST_DEVICE Accessor value(ComponentItemLocalId lid) const
  {
    return Accessor(this->m_value[lid.localId().arrayIndex()].data() + lid.localId().valueIndex());
  }

  //! Positions the value for entity \a item at \a v
  ARCCORE_HOST_DEVICE void setValue(ComponentItemLocalId lid, const DataType& v) const
  {
    this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()] = v;
  }

  ARCCORE_HOST_DEVICE Accessor value0(PureMatVarIndex idx) const
  {
    return Accessor(this->m_value[0][idx.valueIndex()]);
  }

 private:

  ArrayView<DataType>* m_value = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Write view for scalar material variables.
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& vbi, CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell, Accessor>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Write view for scalar material variables
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& vbi, CellEnvironmentVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell, Accessor>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read/write view for scalar material variables
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& vbi, CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell, Accessor>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read/write view for scalar material variables
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& vbi, CellEnvironmentVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell, Accessor>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view for scalar material variables
 */
template <typename DataType> auto
viewIn(const ViewBuildInfo& vbi, const CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell, DataType>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view for scalar material variables
 */
template <typename DataType> auto
viewIn(const ViewBuildInfo& vbi, const CellEnvironmentVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell, DataType>(vbi, var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

#endif
