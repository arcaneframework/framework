// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentSimd.h                                             (C) 2000-2026 */
/*                                                                           */
/* Support for vectorization for materials and media.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_COMPONENTSIMD_H
#define ARCANE_MATERIALS_COMPONENTSIMD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ComponentSimd.h
 *
 * This file contains the different types to manage
 * vectorization on components (materials and media).
 */

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/SimdItem.h"

#include "arcane/materials/MatItem.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __INTEL_COMPILER
#define A_ALIGNED_64 __attribute__((align_value(64)))
#else
#define A_ALIGNED_64
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief SIMD indexer on a component.
 */
class ARCANE_MATERIALS_EXPORT ARCANE_ALIGNAS(64) SimdMatVarIndex
{
 public:

  typedef SimdEnumeratorBase::SimdIndexType SimdIndexType;

 public:

  SimdMatVarIndex(Int32 array_index, SimdIndexType value_index)
  : m_value_index(value_index)
  , m_array_index(array_index)
  {
  }
  SimdMatVarIndex() {}

 public:

  //! Returns the index of the value array in the list of variables.
  Int32 arrayIndex() const { return m_array_index; }

  //! Returns the index in the value array
  const SimdIndexType& valueIndex() const { return m_value_index; }

 private:

  SimdIndexType m_value_index;
  Int32 m_array_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief SIMD enumerator on a sub-part (pure or partial) of a
 * subset of the meshes of a component (material or medium)
 */
class ARCANE_MATERIALS_EXPORT ComponentPartSimdCellEnumerator
: public SimdEnumeratorBase
{
 protected:

  ComponentPartSimdCellEnumerator(IMeshComponent* component, Int32 component_part_index,
                                  Int32ConstArrayView item_indexes)
  : SimdEnumeratorBase(item_indexes)
  , m_component_part_index(component_part_index)
  , m_component(component)
  {
  }

 public:

  static ComponentPartSimdCellEnumerator create(ComponentPartItemVectorView v)
  {
    return ComponentPartSimdCellEnumerator(v.component(), v.componentPartIndex(), v.valueIndexes());
  }

 public:

  SimdMatVarIndex _varIndex() const { return SimdMatVarIndex(m_component_part_index, *_currentSimdIndex()); }

  operator SimdMatVarIndex() const
  {
    return _varIndex();
  }

 protected:

  Integer m_component_part_index;
  IMeshComponent* m_component;
};

inline ComponentPartSimdCellEnumerator
arcaneImplCreateConstituentEnumerator(ComponentPartSimdCell, ComponentPartItemVectorView v)
{
  return ComponentPartSimdCellEnumerator::create(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_SIMD_COMPONENTCELL(iname, env) \
  A_ENUMERATE_COMPONENTCELL(ComponentPartSimdCell, iname, env)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Lambda> void
simple_simd_env_loop(ComponentPartItemVectorView pure_items,
                     ComponentPartItemVectorView impure_items,
                     const Lambda& lambda)
{
  ENUMERATE_COMPONENTITEM (ComponentPartSimdCell, mvi, pure_items) {
    lambda(mvi);
  }
  ENUMERATE_SIMD_COMPONENTCELL(mvi, impure_items)
  {
    lambda(mvi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT LoopFunctorEnvPartSimdCell
{
 public:

  typedef const SimdMatVarIndex& IterType;

 public:

  LoopFunctorEnvPartSimdCell(ComponentPartItemVectorView pure_items,
                             ComponentPartItemVectorView impure_items)
  : m_pure_items(pure_items)
  , m_impure_items(impure_items)
  {}

 public:

  static LoopFunctorEnvPartSimdCell create(const EnvCellVector& env);
  static LoopFunctorEnvPartSimdCell create(IMeshEnvironment* env);

 public:

  template <typename Lambda>
  void operator<<(Lambda&& lambda)
  {
    simple_simd_env_loop(m_pure_items, m_impure_items, lambda);
  }

 private:

  ComponentPartItemVectorView m_pure_items;
  ComponentPartItemVectorView m_impure_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro to iterate over the entities of a component via a
 * C++11 lambda function.
 *
 * The arguments are the same as for the ENUMERATE_COMPONENTITEM() macro.
 *
 * The code after the macro corresponds to the body of the C++11 lambda function.
 * It must therefore be enclosed between two curly braces '{' '}' and end with a
 * semicolon ';'. For example:
 *
 \code
 * ENUMERATE_COMPONENTITEM_LAMBDA(){
 * };
 \endcode
 *
 * \note Even if the code is similar to that of a loop, it is a
 * C++11 lambda function and therefore it is not possible to use keywords like
 * 'break' or 'continue'. If you want to stop an iteration
 * you must use the keyword 'return'.
 */
#define ENUMERATE_COMPONENTITEM_LAMBDA(iter_type, iter, container) \
  Arcane::Materials::LoopFunctor##iter_type ::create((container)) << [=](Arcane::Materials::LoopFunctor##iter_type ::IterType iter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for variable views.
 */
class MatVariableViewBase
{
 public:

  MatVariableViewBase(IMeshMaterialVariable* var)
  : m_variable(var)
  {
  }

 public:

  IMeshMaterialVariable* variable() const { return m_variable; }

 private:

  IMeshMaterialVariable* m_variable;
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
 private:

  typedef MatVarIndex ItemIndexType;
  typedef A_ALIGNED_64 DataType* DataTypeAlignedPtr;

 public:

  MatItemVariableScalarInViewT(IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(var)
  , m_value(v)
  , m_value0(v[0].unguardedBasePointer())
  {}

  //! Vector access operator with indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](const SimdMatVarIndex& mvi) const
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_value[mvi.arrayIndex()].data(), mvi.valueIndex());
  }

  //! Access operator for the \a item entity
  DataType operator[](ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  //! Access operator for the \a item entity
  DataType operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  //! Access operator for the \a item entity
  DataType operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Access operator for the \a item entity
  DataType value(ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  DataType value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

  //! Partial value of the variable for the \a mc iterator
  DataType operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Partial value of the variable for the \a mc iterator
  DataType operator[](EnvCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

 private:

  ArrayView<DataType>* m_value;
  DataTypeAlignedPtr m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view on a scalar mesh variable.
 */
template <typename ItemType, typename DataType>
class MatItemVariableScalarOutViewT
: public MatVariableViewBase
{
 private:

  typedef MatVarIndex ItemIndexType;
  typedef A_ALIGNED_64 DataType* DataTypeAlignedPtr;

 public:

  MatItemVariableScalarOutViewT(IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(var)
  , m_value(v)
  , m_value0(v[0].unguardedBasePointer())
  {}

  //! Vector access operator with indirection.
  SimdSetter<DataType> operator[](const SimdMatVarIndex& mvi) const
  {
    return SimdSetter<DataType>(m_value[mvi.arrayIndex()].data(), mvi.valueIndex());
  }

  //! Access operator for the \a item entity
  DataType& operator[](ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  //! Access operator for the \a item entity
  DataType& operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  DataType& operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Access operator for the \a item entity
  DataType& value(ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  DataType& value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

  //! Partial value of the variable for the \a mc iterator
  DataType& operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Partial value of the variable for the \a mc iterator
  DataType& operator[](EnvCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

 private:

  ArrayView<DataType>* m_value;
  DataTypeAlignedPtr m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read view.
 */
template <typename DataType>
MatItemVariableScalarInViewT<Cell, DataType>
viewIn(const CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell, DataType>(var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Write view
 */
template <typename DataType>
MatItemVariableScalarOutViewT<Cell, DataType>
viewOut(CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarOutViewT<Cell, DataType>(var.materialVariable(), var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
