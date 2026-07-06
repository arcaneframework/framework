// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDVariableViews.h                                           (C) 2000-2026 */
/*                                                                           */
/* multi-dimensional variable view management for accelerators.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_MDVARIABLEVIEWS_H
#define ARCANE_ACCELERATOR_MDVARIABLEVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/VariableViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Read-only view of multi-dimensional matrix variable over a mesh item.
 */
template <typename ItemType_, typename DataType_, int Row, int Column, typename Extents>
class MeshMatrixMDVariableInView
: public VariableViewBase
{
 public:

  using ItemType = ItemType_;
  using DataType = DataType_;
  using ItemLocalIdType = ItemType::LocalIdType;
  using ConstReferenceType = NumMatrixDataViewGetter<DataType, Row, Column>;

  using VariableRefType = MeshMatrixMDVariableRefT<ItemType, DataType, Row, Column, Extents>;

 private:

  using MDSpanType = VariableRefType::MDSpanType;

 public:

  MeshMatrixMDVariableInView(const ViewBuildInfo& view_bi, IVariable* var, const VariableRefType& var_values)
  : VariableViewBase(view_bi, var)
  , m_matrix_mdspan(var_values.m_matrix_mdspan)
  {
  }

 public:

  //! \name Operations for variable of dimension MDDim0
  ///@{

  //! Read-only view of the matrix for item \a id
  constexpr ARCCORE_HOST_DEVICE ConstReferenceType operator()(ItemLocalIdType id) const
  requires(Extents::rank() == 0)
  {
    return ConstReferenceType(m_matrix_mdspan.ptrAt(id.localId()));
  }

  //! Read-only view of the element (i,j) of the matrix for item \a id
  constexpr ARCCORE_HOST_DEVICE DataType operator()(ItemLocalIdType id, Int32 i, Int32 j) const
  requires(Extents::rank() == 0)
  {
    return m_matrix_mdspan(id.localId())(i, j);
  }
  ///@}

  //! \name Operations for variable of dimension MDDim1

  //! Read-only view of the matrix of index \a index for item \a id
  constexpr ARCCORE_HOST_DEVICE ConstReferenceType operator()(ItemLocalIdType id, Int32 index) const
  requires(Extents::rank() == 1)
  {
    return ConstReferenceType(m_matrix_mdspan.ptrAt(id.localId(), index));
  }

  //! Read-only view of the element (i,j) of the matrix for item \a id and index \a index
  constexpr ARCCORE_HOST_DEVICE DataType operator()(ItemLocalIdType id, Int32 index, Int32 i, Int32 j) const
  requires(Extents::rank() == 1)
  {
    return m_matrix_mdspan(id.localId(), index)(i, j);
  }
  ///@}

 private:

  MDSpanType m_matrix_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Read-only view of matrix multi-dimensional mesh variable
 */
template <typename ItemType, typename DataType, int Row, int Column, typename Extents>
auto viewIn(const ViewBuildInfo& command, const MeshMatrixMDVariableRefT<ItemType, DataType, Row, Column, Extents>& var)
{
  IVariable* v = var.underlyingVariable().variable();
  return MeshMatrixMDVariableInView<ItemType, DataType, Row, Column, Extents>(command, v, var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
