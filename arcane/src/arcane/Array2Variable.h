﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Variable.h                                            (C) 2000-2020 */
/*                                                                           */
/* Variable tableau 2D.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ARRAY2VARIABLE_H
#define ARCANE_ARRAY2VARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"

#include "arcane/Variable.h"

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
	
  typedef Array2<T> ValueType;
  typedef IArray2DataT<T> ValueDataType;
  typedef Array2VariableT<T> ThatClass;
  typedef Variable BaseClass;

 protected:

  //! Construit une variable basée sur la référence \a v
  Array2VariableT(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  static ARCANE_CORE_EXPORT ThatClass* getReference(IVariable* var);
  static ARCANE_CORE_EXPORT ThatClass* getReference(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  Integer checkIfSame(IDataReader* reader,int max_print,bool compare_ghost) override;
  void synchronize() override;
  Real allocatedMemory() const override;
  Integer checkIfSync(int max_print) override;
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

 public:

  ARCANE_CORE_EXPORT void directResize(Integer dim1);
  ARCANE_CORE_EXPORT void directResize(Integer dim1,Integer dim2);
  ARCANE_CORE_EXPORT void swapValues(ThatClass& rhs);
  ValueDataType* trueData() { return m_data; }

 protected:
  
  void _internalResize(Integer new_size,Integer added_memory) override;
  Integer _checkIfSameOnAllReplica(IParallelMng* replica_pm,int max_print) override;

 private:

  ValueDataType* m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

