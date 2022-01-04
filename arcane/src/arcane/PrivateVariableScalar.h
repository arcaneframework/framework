// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableScalar.h                                     (C) 2000-2020 */
/*                                                                           */
/* Classe gérant une variable sur une entité du maillage.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PRIVATEVARIABLESCALAR_H
#define ARCANE_PRIVATEVARIABLESCALAR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshVariableRef.h"

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
#ifdef ARCANE_PROXY
  typedef typename DataTypeTraitsT<DataType>::ProxyType ProxyType;
#endif
  
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
  
 protected:
  
  void _internalInit() { MeshVariableRef::_internalInit(m_private_part); }

#ifdef ARCANE_TRACE
private:
  
  typedef ArrayView<DataType> ArrayBase;
  void _trace(Integer local_id) const 
  {
    if (m_has_trace){
      IDataTracerT<DataType>* tracer = m_trace_infos[local_id];
      if (tracer)
        tracer->traceRead(ArrayBase::operator[](local_id));
    }
  }
#endif
  
protected:
  
  const DataType& _value(Integer local_id) const
  {
#ifdef ARCANE_TRACE
    _trace(local_id);
#endif
    return m_view[local_id];
  }
  
  DataTypeReturnReference _value(Integer local_id)
  {
#ifdef ARCANE_TRACE
    _trace(local_id);
#endif
    return m_view[local_id];
  }
  
#ifdef ARCANE_PROXY
  ProxyType _getProxy(Integer local_id)
  {
    return ProxyType(_value(local_id),_getMemoryInfo(local_id));
  }
  MemoryAccessInfo _getMemoryInfo(Integer local_id) const
  {
    return MemoryAccessInfo(&m_access_infos[local_id],m_memory_trace,local_id);
  }
#endif
  
  const DataType& _putValue(Integer index,const DataType& v)
  {
    return (_value(index) = v);
  }
  
protected:

  PrivatePartType* m_private_part;
    
  ArrayView<DataType> m_view;
  
  IMemoryAccessTrace* m_memory_trace;
  
#ifdef ARCANE_TRACE    
  mutable ArrayView<Byte> m_access_infos;
  
  ConstArrayView< IDataTracerT<DataType>* > m_trace_infos;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
