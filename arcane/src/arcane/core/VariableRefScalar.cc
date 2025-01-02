// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefScalar.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Référence à une variable scalaire.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRefScalar.h"
#include "arcane/core/VariableScalar.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/VariableFactoryRegisterer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableFactoryRegisterer
VariableRefScalarT<DataType>::
m_auto_registerer(_autoCreate,_buildVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableTypeInfo
VariableRefScalarT<DataType>::
_buildVariableTypeInfo()
{
  return VariableTypeInfo(IK_Unknown,VariableDataTypeTraitsT<DataType>::type(),0,0,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
VariableRefScalarT<DataType>::
_buildVariableInfo(const VariableBuildInfo& vbi)
{
  VariableTypeInfo vti = _buildVariableTypeInfo();
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableRef*
VariableRefScalarT<DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> 
VariableRefScalarT<DataType>::
VariableRefScalarT(const VariableBuildInfo& vb)
: VariableRef(vb)
, m_private_part(PrivatePartType::getReference(vb,_buildVariableInfo(vb)))
{
  _internalInit(m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> 
VariableRefScalarT<DataType>::
VariableRefScalarT(IVariable* var)
: VariableRef(var)
, m_private_part(PrivatePartType::getReference(var))
{
  _internalInit(m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
VariableRefScalarT<DataType>::
VariableRefScalarT(const VariableRefScalarT<DataType>& rhs)
: VariableRef(rhs)
, m_private_part(rhs.m_private_part)
{
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void 
VariableRefScalarT<DataType>::
refersTo(const VariableRefScalarT<DataType>& rhs)
{
  VariableRef::operator=(rhs);
  m_private_part = rhs.m_private_part;
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void 
VariableRefScalarT<DataType>::
assign(const DataType& v)
{
  m_private_part->value() = v;
  m_private_part->syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<typename T>
void _reduce(T& value,IParallelMng* pm,IParallelMng::eReduceType t,FalseType)
{
  ARCANE_UNUSED(value);
  ARCANE_UNUSED(pm);
  ARCANE_UNUSED(t);
}

template<typename T>
void _reduce(T& value,IParallelMng* pm,IParallelMng::eReduceType t,TrueType)
{
  T r = value;
  value = pm->reduce(t,r);
}
}


template<typename DataType> void 
VariableRefScalarT<DataType>::
reduce(IParallelMng::eReduceType type)
{
  typedef typename VariableDataTypeTraitsT<ElementType>::HasReduce  HasReduce;
  ElementType v = m_private_part->value();
  _reduce(v,variableMng()->parallelMng(),type,HasReduce());
  assign(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void 
VariableRefScalarT<T>::
updateFromInternal()
{
  BaseClass::updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange les valeurs de la variable \a rhs avec celles de l'instance.
 */
template<typename DataType> void
VariableRefScalarT<DataType>::
swapValues(VariableRefScalarT<DataType>& rhs)
{
  this->m_private_part->swapValues(*(rhs.m_private_part));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(VariableRefScalarT);
template class VariableRefScalarT<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
