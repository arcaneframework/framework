// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableArray.h                                             (C) 2000-2025 */
/*                                                                           */
/* Variable tableau 1D.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEARRAY_H
#define ARCANE_CORE_VARIABLEARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Variable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'une variable sur un tableau 1D.
 *
 * Les méthodes de cette classe sont internes à %Arcane.
 */
template<class T>
class VariableArrayT
: public Variable
{
 public:
	
  typedef Array<T> ValueType;
  typedef IArrayDataT<T> ValueDataType;
  typedef VariableArrayT<T> ThatClass;
  typedef Variable BaseClass;

 protected:

  //! Construit une variable basée sur la référence \a v
  VariableArrayT(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  ~VariableArrayT() override;

 public:

  static ARCANE_CORE_EXPORT ThatClass* getReference(IVariable* var);
  static ARCANE_CORE_EXPORT ThatClass* getReference(const VariableBuildInfo& v,const VariableInfo& vi);
                                   
 public:

  void synchronize() override;
  void synchronize(Int32ConstArrayView local_ids) override;
  virtual void resizeWithReserve(Integer n,Integer nb_additional);
  Real allocatedMemory() const override;
  bool initialize(const ItemGroup& group,const String& value) override;
  Integer nbElement() const override { return m_value->view().size(); }
  ARCCORE_DEPRECATED_2021("use valueView() instead")
  ARCANE_CORE_EXPORT ValueType& value();
  ConstArrayView<T> constValueView() const { return m_value->view(); }
  ConstArrayView<T> valueView() const { return m_value->view(); }
  ArrayView<T> valueView() { return m_value->view(); }
  ARCANE_CORE_EXPORT void shrinkMemory() override;
  ARCANE_CORE_EXPORT Integer capacity();
  void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) override;
  void copyItemsMeanValues(Int32ConstArrayView first_source,
                           Int32ConstArrayView second_source,
                           Int32ConstArrayView destination) override;
  void compact(Int32ConstArrayView old_to_new_ids) override;
  void print(std::ostream& o) const override;
  void setIsSynchronized() override;
  void setIsSynchronized(const ItemGroup& item_group) override;
  IData* data() override { return m_value; }
  const IData* data() const override { return m_value; }

  virtual void fill(const T& v);
  virtual void fill(const T& v,const ItemGroup& item_group);

 public:

  ARCANE_CORE_EXPORT void swapValues(ThatClass& rhs);
  ValueDataType* trueData() { return m_value; }

 protected:

  void _internalResize(const VariableResizeArgs& resize_args) override;
  Integer _checkIfSameOnAllReplica(IParallelMng* replica_pm,int max_print) override;
  VariableComparerResults _compareVariable(const VariableComparerArgs& compare_args) final;

 private:

  ValueDataType* m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
