// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMDVariableRef.h                                         (C) 2000-2026 */
/*                                                                           */
/* Class managing a multi-dimensional variable on a mesh entity.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHMDVARIABLEREF_H
#define ARCANE_CORE_MESHMDVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayLayout.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MDSpan.h"

#include "arcane/core/DataView.h"

#include "arcane/core/MeshVariableArrayRef.h"
#include "arcane/core/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * ATTENTION:
 *
 * All classes in this file are experimental and the API is not
 * fixed. DO NOT USE OUTSIDE OF ARCANE.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType>
class MeshMDVariableRefWrapperT
: public MeshVariableArrayRefT<ItemType, DataType>
{
  template <typename _ItemType, typename _DataType, typename _Extents>
  friend class Arcane::MeshMDVariableRefBaseT;

 public:

  using BaseClass = MeshVariableArrayRefT<ItemType, DataType>;
  using VariableType = typename BaseClass::PrivatePartType;
  using ValueDataType = typename VariableType::ValueDataType;

 private:

  explicit MeshMDVariableRefWrapperT(const VariableBuildInfo& vbi)
  : BaseClass(vbi)
  {
  }

 private:

  ValueDataType* trueData() { return this->m_private_part->trueData(); }
  const ValueDataType* trueData() const { return this->m_private_part->trueData(); }

  void fillShape(ArrayShape& shape_with_item)
  {
    this->m_private_part->fillShape(shape_with_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class managing a multi-dimensional variable on a mesh entity.
 *
 * \warning API is under definition. Do not use outside of Arcane.
 */
template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefBaseT
: public MeshVariableRef
{
 public:

  using UnderlyingVariableType = MeshVariableArrayRefT<ItemType, DataType>;
  using MDSpanType = MDSpan<DataType, Extents, RightLayout>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  using FullExtentsType = Extents;

 public:

  explicit MeshMDVariableRefBaseT(const VariableBuildInfo& b)
  : MeshVariableRef(b)
  , m_underlying_var(b)
  {
    _internalInit(m_underlying_var.variable());
  }

  //! Associated underlying variable.
  UnderlyingVariableType& underlyingVariable() { return m_underlying_var; }

  //! Full shape (static + dynamic) of the variable.
  ArrayShape fullShape() const { return m_underlying_var.trueData()->shape(); }

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

  impl::MeshMDVariableRefWrapperT<ItemType, DataType> m_underlying_var;
  MDSpanType m_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a multi-dimensional variable on a mesh entity.
 *
 * \warning API is under definition. Do not use outside of Arcane.
 */
template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType, typename Extents::template AddedFirstExtentsType<DynExtent>>
{
  using AddedFirstExtentsType = typename Extents::template AddedFirstExtentsType<DynExtent>;
  using BasicType = typename DataTypeTraitsT<DataType>::BasicType;
  static_assert(Extents::rank() >= 0 && Extents::rank() <= 3, "Only Extents of rank 0, 1, 2 or 3 are implemented");
  static_assert(std::is_same_v<DataType,BasicType>,"DataType should be a basic type (Real, Int32, Int64, ... )");

 public:

  using BaseClass = MeshMDVariableRefBaseT<ItemType, DataType, AddedFirstExtentsType>;
  using ItemLocalIdType = typename ItemType::LocalIdType;
  static constexpr int nb_dynamic = Extents::nb_dynamic;

 public:

  explicit MeshMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  DataType& operator()(ItemLocalIdType id)
  {
    return this->m_mdspan(id.localId());
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  const DataType& operator()(ItemLocalIdType id) const
  {
    return this->m_mdspan(id.localId());
  }


  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  DataType& operator()(ItemLocalIdType id, Int32 i1)
  {
    return this->m_mdspan(id.localId(), i1);
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  const DataType& operator()(ItemLocalIdType id, Int32 i1) const
  {
    return this->m_mdspan(id.localId(), i1);
  }

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

  /*!
   * \brief Changes the data shape.
   *
   * The number of elements in \a dims must correspond to the number of dynamic values
   * in \a Extents.
   */
  void reshape(std::array<Int32, Extents::nb_dynamic> dims)
  {
    ArrayShape shape(dims);
    this->m_underlying_var.resizeAndReshape(shape);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a multi-dimensional 'NumVector' type variable on a mesh entity.
 *
 * \warning API is under definition. Do not use outside of Arcane.
 */
template <typename ItemType, typename DataType, int Size, typename Extents>
class MeshVectorMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType, typename Extents::template AddedFirstLastExtentsType<DynExtent, Size>>
{
 public:

  using NumVectorType = NumVector<DataType, Size>;

 private:

  using BasicType = typename DataTypeTraitsT<DataType>::BasicType;
  using AddedFirstLastExtentsType = typename Extents::template AddedFirstLastExtentsType<DynExtent, Size>;
  using AddedFirstExtentsType = typename Extents::template AddedFirstExtentsType<DynExtent>;
  using BaseClass = MeshMDVariableRefBaseT<ItemType, DataType, AddedFirstLastExtentsType>;
  static_assert(Extents::rank() >= 0 && Extents::rank() <= 2, "Only Extents of rank 0, 1 or 2 are implemented");
  static_assert(std::is_same_v<DataType, BasicType>, "DataType should be a basic type (Real, Int32, Int64, ... )");

 public:

  using ItemLocalIdType = typename ItemType::LocalIdType;
  using ReferenceType = DataViewGetterSetter<NumVectorType>;
  using ConstReferenceType = DataViewGetter<NumVectorType>;
  using MDSpanType = MDSpan<NumVectorType, AddedFirstExtentsType, RightLayout>;
  static constexpr int nb_dynamic = Extents::nb_dynamic;

 public:

  explicit MeshVectorMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  //! Accesses the data for reading/writing
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  ReferenceType operator()(ItemLocalIdType id)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId()));
  }

  //! Accesses the data for reading
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  ConstReferenceType operator()(ItemLocalIdType id) const
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId()));
  }

  //! Accesses the data for reading/writing
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  ReferenceType operator()(ItemLocalIdType id, Int32 i1)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1));
  }

  //! Accesses the data for reading
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  ConstReferenceType operator()(ItemLocalIdType id, Int32 i1) const
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1));
  }

  //! Accesses the data for reading/writing
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  ReferenceType operator()(ItemLocalIdType id, Int32 i1, Int32 i2)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1, i2));
  }

  //! Accesses the data for reading
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  ConstReferenceType operator()(ItemLocalIdType id, Int32 i1, Int32 i2) const
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1, i2));
  }

  /*!
   * \brief Changes the data shape.
   *
   * The number of elements in \a dims must correspond to the number of dynamic values
   * in \a Extents.
   */
  void reshape(std::array<Int32, Extents::nb_dynamic> dims)
  {
    std::array<Int32, nb_dynamic + 1> full_dims;
    // We add 'Size' to the end of the dimensions.
    for (int i = 0; i < nb_dynamic; ++i)
      full_dims[i] = dims[i];
    full_dims[nb_dynamic] = Size;
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
    // from 'DataType' to 'NumVector<DataType,Size>'.
    DataType* v = this->m_mdspan.to1DSpan().data();
    NumVectorType* nv = reinterpret_cast<NumVectorType*>(v);
    m_vector_mdspan = MDSpanType(nv, this->m_mdspan.extents().dynamicExtents());
  }

 private:

  MDSpanType m_vector_mdspan;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing a multi-dimensional 'NumMatrix' type variable on a
 * mesh entity.
 *
 * \warning API is under definition. Do not use outside of Arcane.
 */
template <typename ItemType, typename DataType, int Row, int Column, typename Extents>
class MeshMatrixMDVariableRefT
: public MeshMDVariableRefBaseT<ItemType, DataType, typename Extents::template AddedFirstLastLastExtentsType<DynExtent, Row, Column>>
{
 public:

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
  using ReferenceType = DataViewGetterSetter<NumMatrixType>;
  using ConstReferenceType = DataViewGetter<NumMatrixType>;
  using MDSpanType = MDSpan<NumMatrixType, AddedFirstExtentsType, RightLayout>;
  static constexpr int nb_dynamic = Extents::nb_dynamic;

 public:

  explicit MeshMatrixMDVariableRefT(const VariableBuildInfo& b)
  : BaseClass(b)
  {}

 public:

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  ReferenceType operator()(ItemLocalIdType id)
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId()));
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 0, void>>
  ConstReferenceType operator()(ItemLocalIdType id) const
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId()));
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  ReferenceType operator()(ItemLocalIdType id, Int32 i1)
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId(), i1));
  }

  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  ConstReferenceType operator()(ItemLocalIdType id, Int32 i1) const
  {
    return ReferenceType(m_matrix_mdspan.ptrAt(id.localId(), i1));
  }

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
