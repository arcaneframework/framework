// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray2.cc                                        (C) 2000-2021 */
/*                                                                           */
/* Classe gérant une référence sur une variable tableau 2D.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableRefArray2.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/Array2Variable.h"
#include "arcane/VariableBuildInfo.h"
#include "arcane/VariableInfo.h"
#include "arcane/VariableDataTypeTraits.h"
#include "arcane/IParallelMng.h"
#include "arcane/VariableFactoryRegisterer.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> VariableFactoryRegisterer
VariableRefArray2T<DataType>::
m_auto_registerer(_autoCreate,_internalVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableTypeInfo
VariableRefArray2T<DataType>::
_internalVariableTypeInfo()
{
  return VariableTypeInfo(IK_Unknown,VariableDataTypeTraitsT<DataType>::type(),2,0,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
VariableRefArray2T<DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi)
{
  VariableTypeInfo vti = _internalVariableTypeInfo();
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> VariableRef*
VariableRefArray2T<DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> 
VariableRefArray2T<DataType>::
VariableRefArray2T(const VariableBuildInfo& vbi)
: VariableRef(vbi)
, m_private_part(PrivatePartType::getReference(vbi,_internalVariableInfo(vbi)))
{
  _internalInit(m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
VariableRefArray2T<DataType>::
VariableRefArray2T(const VariableRefArray2T<DataType>& rhs)
: VariableRef(rhs)
, Array2View<DataType>(rhs)
, m_private_part(rhs.m_private_part)
{
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> VariableRefArray2T<DataType>::
VariableRefArray2T(IVariable* var)
: VariableRef(var)
, m_private_part(PrivatePartType::getReference(var))
{
  this->_internalInit(this->m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
VariableRefArray2T<DataType>::
operator=(const VariableRefArray2T<DataType>& rhs)
{
  VariableRef::operator=(rhs);
  m_private_part = rhs.m_private_part;
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableRefArray2T<DataType>::
~VariableRefArray2T()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
VariableRefArray2T<DataType>::
refersTo(const VariableRefArray2T<DataType>& rhs)
{
  VariableRef::operator=(rhs);
  m_private_part = rhs.m_private_part;
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
VariableRefArray2T<DataType>::
resize(Integer s)
{
  m_private_part->directResize(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
VariableRefArray2T<DataType>::
resize(Integer dim1_size,Integer dim2_size)
{
  m_private_part->directResize(dim1_size,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
VariableRefArray2T<DataType>::
fill(const DataType& value)
{
  m_private_part->trueData()->_internal()->_internalDeprecatedValue().fill(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void VariableRefArray2T<DataType>::
updateFromInternal()
{
  ArrayBase::operator=(m_private_part->valueView());
  BaseClass::updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> auto VariableRefArray2T<T>::
internalContainer() -> ContainerType&
{
  return _internalTrueData()->_internalDeprecatedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> IArray2DataInternalT<T>*
VariableRefArray2T<T>::
_internalTrueData()
{
  if (!(property() & IVariable::PPrivate))
    ARCANE_FATAL("variable '{0}': getting internal data container is only valid on private variable", name());
  return m_private_part->trueData()->_internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class VariableRefArray2T<Byte>;
template class VariableRefArray2T<Real>;
template class VariableRefArray2T<Int16>;
template class VariableRefArray2T<Int32>;
template class VariableRefArray2T<Int64>;
template class VariableRefArray2T<Real2>;
template class VariableRefArray2T<Real3>;
template class VariableRefArray2T<Real2x2>;
template class VariableRefArray2T<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
