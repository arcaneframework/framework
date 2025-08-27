// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArrayLock.h                                      (C) 2000-2021 */
/*                                                                           */
/* Verrou sur une variable tableau.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEREFARRAYLOCK_H
#define ARCANE_VARIABLEREFARRAYLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/IVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Verrou sur une variable tableau.
 * \deprecated Ne plus utiliser
 */
template<typename DataType>
class VariableRefArrayLockT
{
 public:

  typedef Array<DataType> ValueType;
  typedef VariableRefArrayLockT<DataType> ThatClass;

 public:

  VariableRefArrayLockT(ValueType& v,IVariable* var)
  : m_value(v), m_variable(var), m_saved_ptr(v.data()), m_saved_size(v.size())
  {
  }

  ~VariableRefArrayLockT()
  {
    if (m_value.data()!=m_saved_ptr || m_value.size()!=m_saved_size)
      m_variable->syncReferences();
  }

 public:

  VariableRefArrayLockT(const VariableRefArrayLockT<DataType>& rhs) = default;
  ThatClass& operator=(const ThatClass& rhs) = default;

 public:

  ARCCORE_DEPRECATED_2021("This class is deprecated")
  ValueType& value() { return m_value; }

 private:

  ValueType& m_value;
  IVariable* m_variable;
  DataType* m_saved_ptr;
  Integer m_saved_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
