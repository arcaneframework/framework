// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMatrixMDVariableRef.h                                   (C) 2000-2026 */
/*                                                                           */
/* Multi-dimensional 'NumMatrix' variable on a mesh entity.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHMATRIXMDVARIABLEREF_H
#define ARCANE_CORE_MESHMATRIXMDVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumMatrix.h"
#include "arcane/utils/NumMatrixDataView.h"

#include "arcane/core/MeshMDVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class managing a multi-dimensional NumMatrix type variable on a
 * mesh entity.
 *
 * The dimension of the matrix is fixed and is given by (Row,Column).
 *
 * \warning You need to call the method reshape() before using this kind
 * of variables.
 *
 * For more information, see \ref arcanedoc_core_types_axl_md_variable_use.
 */
template <typename ItemType, typename DataType_, int Row, int Column, typename Extents>
class MeshMatrixMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType_, typename Extents::template AddedFirstLastLastExtentsType<DynExtent, Row, Column>>
{
  // To access m_matrix_mdspan
  friend class Arcane::Accelerator::MeshMatrixMDVariableInOutView<ItemType, DataType_, Row, Column, Extents>;
  friend class Arcane::Accelerator::MeshMatrixMDVariableInView<ItemType, DataType_, Row, Column, Extents>;

 public:

  using DataType = DataType_;
  using NumMatrixType = NumMatrix<DataType, Row, Column>;

 private:

  using BasicType = typename DataTypeTraitsT<DataType>::BasicType;
  using AddedFirstLastLastExtentsType = typename Extents::template AddedFirstLastLastExtentsType<DynExtent, Row, Column>;
  using AddedFirstExtentsType = typename Extents::template AddedFirstExtentsType<DynExtent>;
  using BaseClass = MeshMDVariableRefBaseT<ItemType, DataType, AddedFirstLastLastExtentsType>;
  static_assert(Extents::rank() >= 0 && Extents::rank() <= 1, "Only Extents of rank 0 or 1 are implemented");
  static_assert(std::is_same_v<DataType, BasicType>, "DataType should be a basic type (Real, Int32, Int64, ... )");

 public:

  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ReferenceType = NumMatrixDataViewGetterSetter<DataType, Row, Column>;
  using ConstReferenceType = NumMatrixDataViewGetter<DataType, Row, Column>;
  using MDSpanType = MDSpan<NumMatrixType, AddedFirstExtentsType, RightLayout>;
  static constexpr int nb_dynamic = Extents::nb_dynamic;

 public:

  explicit MeshMatrixMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  //! \name Operations for variable of dimension MDDim0
  ///@{
  //! Mutable view of the matrix for item \a id
  ReferenceType operator()(ItemLocalIdType id) requires(Extents::rank() == 0)
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId()));
  }

  //! Read-only view of the matrix for item \a id
  ConstReferenceType operator()(ItemLocalIdType id) const requires(Extents::rank() == 0)
  {
    return ConstReferenceType(m_matrix_mdspan.ptrAt(id.localId()));
  }

  //! Mutable view of the element (i,j) of the matrix for item \a id
  DataType& operator()(ItemLocalIdType id, Int32 i, Int32 j) requires(Extents::rank() == 0)
  {
    return m_matrix_mdspan(id.localId())(i, j);
  }

  //! Read-only view of the element (i,j) of the matrix for item \a id
  DataType operator()(ItemLocalIdType id, Int32 i, Int32 j) const requires(Extents::rank() == 0)
  {
    return m_matrix_mdspan(id.localId())(i, j);
  }
  ///@}

  //! \name Operations for variable of dimension MDDim1
  ///@{
  //! Mutable view of the matrix of index \a index for item \a id
  ReferenceType operator()(ItemLocalIdType id, Int32 index)
  requires(Extents::rank() == 1)
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId(), index));
  }

  //! Read-only view of the matrix of index \a index for item \a id
  ConstReferenceType operator()(ItemLocalIdType id, Int32 index) const
  requires(Extents::rank() == 1)
  {
    return ConstReferenceType(m_matrix_mdspan.ptrAt(id.localId(), index));
  }

  //! Mutable view of the element (i,j) of the matrix for item \a id and index \a index
  DataType& operator()(ItemLocalIdType id, Int32 index, Int32 i, Int32 j)
  requires(Extents::rank() == 1)
  {
    return m_matrix_mdspan(id.localId(), index)(i, j);
  }

  //! Read-only view of the element (i,j) of the matrix for item \a id and index \a index
  DataType operator()(ItemLocalIdType id, Int32 index, Int32 i, Int32 j) const
  requires(Extents::rank() == 1)
  {
    return m_matrix_mdspan(id.localId(), index)(i, j);
  }
  ///@}

  /*!
   * \brief Changes the data shape.
   *
   * The number of elements in \a dims must correspond to the number of dynamic values
   * in \a Extents.
   */
  void reshape(std::array<Int32, Extents::nb_dynamic> dims)
  {
    std::array<Int32, nb_dynamic + 2> full_dims;
    // We add 'Row' and 'Column' to the end of the dimensions.
    for (int i = 0; i < nb_dynamic; ++i)
      full_dims[i] = dims[i];
    full_dims[nb_dynamic] = Row;
    full_dims[nb_dynamic + 1] = Column;
    ArrayShape shape(full_dims);
    this->m_underlying_var.resizeAndReshape(shape);
  }

 protected:

  void updateFromInternal() override
  {
    BaseClass::updateFromInternal();
    // Positions the value of m_vector_mdspan.
    // It will have the same dimensions as m_mdspan except that we
    // remove the last dimension and change the type
    // from 'DataType' to 'NumMatrix<DataType,Row,Column>'.
    DataType* v = this->m_mdspan.to1DSpan().data();
    NumMatrixType* nv = reinterpret_cast<NumMatrixType*>(v);
    m_matrix_mdspan = MDSpanType(nv, this->m_mdspan.extents().dynamicExtents());
  }

 private:

  MDSpanType m_matrix_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
