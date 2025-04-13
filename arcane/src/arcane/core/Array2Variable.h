// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Variable.h                                            (C) 2000-2025 */
/*                                                                           */
/* Variable tableau 2D.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ARRAY2VARIABLE_H
#define ARCANE_CORE_ARRAY2VARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"

#include "arcane/core/Variable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Variable sur un tableau 2D.
 */
template<class T>
class Array2VariableT
: public Variable
{
 public:

  using ValueType = Array2<T>;
  using ValueDataType = IArray2DataT<T>;
  using ThatClass = Array2VariableT<T>;
  using BaseClass = Variable;

 protected:

  //! Construit une variable basée sur la référence \a v
  Array2VariableT(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  static ARCANE_CORE_EXPORT ThatClass* getReference(IVariable* var);
  static ARCANE_CORE_EXPORT ThatClass* getReference(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  void synchronize() override;
  void synchronize(Int32ConstArrayView local_ids) override;
  Real allocatedMemory() const override;
  Integer nbElement() const override { return m_data->view().totalNbElement(); }
  ARCCORE_DEPRECATED_2021("Use valueView() instead")
  virtual ValueType& value();
  ConstArray2View<T> constValueView() const { return m_data->view(); }
  ConstArray2View<T> valueView() const { return m_data->view(); }
  Array2View<T> valueView() { return m_data->view(); }
  void shrinkMemory() override;
  void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) override;
  void copyItemsMeanValues(Int32ConstArrayView first_source,
                           Int32ConstArrayView second_source,
                           Int32ConstArrayView destination) override;
  void compact(Int32ConstArrayView old_to_new_ids) override;
  void print(std::ostream& o) const override;
  void setIsSynchronized() override;
  void setIsSynchronized(const ItemGroup& item_group) override;
  IData* data() override { return m_data; }
  const IData* data() const override { return m_data; }

 public:

  ARCANE_CORE_EXPORT void directResize(Integer dim1);
  ARCANE_CORE_EXPORT void directResize(Integer dim1,Integer dim2);
  ARCANE_CORE_EXPORT void directResizeAndReshape(const ArrayShape& shape);
  ARCANE_CORE_EXPORT void swapValues(ThatClass& rhs);
  ARCANE_CORE_EXPORT void fillShape(ArrayShape& shape);
  ValueDataType* trueData() { return m_data; }

 protected:
  
  void _internalResize(const VariableResizeArgs& resize_args) override;
  VariableComparerResults _compareVariable(const VariableComparerArgs& compare_args) final;

 private:

  ValueDataType* m_data = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

