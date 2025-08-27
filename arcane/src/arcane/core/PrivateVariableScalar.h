// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableScalar.h                                     (C) 2000-2024 */
/*                                                                           */
/* Classe gérant une variable sur une entité du maillage.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PRIVATEVARIABLESCALAR_H
#define ARCANE_PRIVATEVARIABLESCALAR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Classe de factorisation des variables scalaires sur des entités du maillage.
 */
template<typename DataType>
class PrivateVariableScalarT
: public MeshVariableRef
{
 protected:
  
  typedef DataType& DataTypeReturnReference;
  typedef VariableArrayT<DataType> PrivatePartType;
  
 protected:
  
  //! Construit une référence liée au module \a module
  ARCANE_CORE_EXPORT PrivateVariableScalarT(const VariableBuildInfo& vb, const VariableInfo& vi);
  
 protected:
  
  ARCANE_CORE_EXPORT PrivateVariableScalarT(const PrivateVariableScalarT& rhs);
  ARCANE_CORE_EXPORT PrivateVariableScalarT(IVariable* var);
  ARCANE_CORE_EXPORT PrivateVariableScalarT();
  
  ARCANE_CORE_EXPORT void operator=(const PrivateVariableScalarT& rhs);

 public:
  
  ArrayView<DataType> asArray(){ return m_view; }
  ConstArrayView<DataType> asArray() const { return m_view; }
  Integer arraySize() const { return 0; }
  
  ARCANE_CORE_EXPORT void updateFromInternal();

  ARCANE_CORE_EXPORT ItemGroup itemGroup() const;
  
 public:

  SmallSpan<DataType> _internalSpan() { return m_view; }
  SmallSpan<const DataType> _internalSpan() const { return m_view; }
  SmallSpan<const DataType> _internalConstSpan() const { return m_view; }

 protected:
  
  void _internalInit() { MeshVariableRef::_internalInit(m_private_part); }
  
 protected:
  
  const DataType& _value(Integer local_id) const { return m_view[local_id]; }
  DataTypeReturnReference _value(Integer local_id) { return m_view[local_id]; }
  
  const DataType& _putValue(Integer index,const DataType& v)
  {
    return (_value(index) = v);
  }
  
 protected:

  PrivatePartType* m_private_part = nullptr;

  ArrayView<DataType> m_view;

  IMemoryAccessTrace* m_memory_trace = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
