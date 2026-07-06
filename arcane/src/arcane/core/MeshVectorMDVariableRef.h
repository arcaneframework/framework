// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVectorMDVariableRef.h                                   (C) 2000-2026 */
/*                                                                           */
/* Multi-dimensional 'NumVector' variable on a mesh entity.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHVECTORMDVARIABLEREF_H
#define ARCANE_CORE_MESHVECTORMDVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumVector.h"

#include "arcane/core/DataView.h"
#include "arcane/core/MeshMDVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class managing a multi-dimensional NumVector type variable on a mesh entity.
 *
 * \warning You need to call the method reshape() before using this kind
 * of variables.
 *
 * For more information, see \ref arcanedoc_core_types_axl_md_variable_use.
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

  //! \name Operations for variable of dimension MDDim0
  ///@{
  //! Accesses the data for reading/writing
  ReferenceType operator()(ItemLocalIdType id)
  requires(Extents::rank() == 0)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId()));
  }

  //! Accesses the data for reading
  ConstReferenceType operator()(ItemLocalIdType id) const
  requires(Extents::rank() == 0)
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId()));
  }
  ///@}

  //! \name Operations for variable of dimension MDDim1
  ///@{
  //! Accesses the data for reading/writing
  ReferenceType operator()(ItemLocalIdType id, Int32 i1)
  requires(Extents::rank() == 1)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1));
  }

  //! Accesses the data for reading
  ConstReferenceType operator()(ItemLocalIdType id, Int32 i1) const
  requires(Extents::rank() == 1)
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1));
  }
  ///@}

  //! \name Operations for variable of dimension MDDim2
  ///@{
  //! Accesses the data for reading/writing
  ReferenceType operator()(ItemLocalIdType id, Int32 i1, Int32 i2)
  requires(Extents::rank() == 2)
  {
    return ReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1, i2));
  }

  //! Accesses the data for reading
  ConstReferenceType operator()(ItemLocalIdType id, Int32 i1, Int32 i2) const
  requires(Extents::rank() == 2)
  {
    return ConstReferenceType(m_vector_mdspan.ptrAt(id.localId(), i1, i2));
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
