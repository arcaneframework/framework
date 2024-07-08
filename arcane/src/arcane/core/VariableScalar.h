﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableScalar.h                                            (C) 2000-2024 */
/*                                                                           */
/* Variable scalaire.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLE_SCALAR_H
#define ARCANE_VARIABLE_SCALAR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Variable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Variable scalaire.
 */
template<class T>
class VariableScalarT
: public Variable
{
 public:
	
  typedef T ValueType;
  typedef IScalarDataT<T> ValueDataType;
  typedef VariableScalarT<T> ThatClass;
  typedef Variable BaseClass;

 protected:

  //! Construit une variable basée sur la référence \a v
  VariableScalarT(const VariableBuildInfo& v,const VariableInfo& vi);

 public:

  static ARCANE_CORE_EXPORT ThatClass* getReference(const VariableBuildInfo& v,const VariableInfo& vi);
  static ARCANE_CORE_EXPORT ThatClass* getReference(IVariable* var);

 public:

  Integer checkIfSame(IDataReader* reader,int max_print,bool compare_ghost) override;
  void synchronize() override;
  void synchronize(Int32ConstArrayView local_ids) override;
  Real allocatedMemory() const override;
  Integer nbElement() const override { return 1; }
  ValueType& value() { return m_value->value(); }
  void shrinkMemory() override { }
  void print(std::ostream& o) const override;
  IData* data() override { return m_value; }
  const IData* data() const override { return m_value; }

 public:
  
  void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) override;
  void copyItemsMeanValues(Int32ConstArrayView first_source,
                           Int32ConstArrayView second_source,
                           Int32ConstArrayView destination) override;
  void compact(Int32ConstArrayView new_to_old_ids) override;
  void setIsSynchronized() override;
  void setIsSynchronized(const ItemGroup& item_group) override;

 public:

  ARCANE_CORE_EXPORT void swapValues(ThatClass& rhs);

 protected:

  void _internalResize(Integer new_size,Integer nb_additional_element) override
  {
    ARCANE_UNUSED(new_size);
    ARCANE_UNUSED(nb_additional_element);
  }
  Integer _checkIfSameOnAllReplica(IParallelMng* replica_pm,int max_print) override;

 private:

  ValueDataType* m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
