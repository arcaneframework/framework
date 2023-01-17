﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Référence à une variable tableau 1D.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableRefArray.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/VariableArray.h"
#include "arcane/VariableRefArrayLock.h"
#include "arcane/VariableBuildInfo.h"
#include "arcane/VariableInfo.h"
#include "arcane/VariableDataTypeTraits.h"
#include "arcane/ISubDomain.h"
#include "arcane/VariableFactoryRegisterer.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
VariableFactoryRegisterer
VariableRefArrayT<DataType>::
m_auto_registerer(_autoCreate, _internalVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
VariableTypeInfo
VariableRefArrayT<DataType>::
_internalVariableTypeInfo()
{
  return VariableTypeInfo(IK_Unknown, VariableDataTypeTraitsT<DataType>::type(), 1, 0, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
VariableInfo
VariableRefArrayT<DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi)
{
  VariableTypeInfo vti = _internalVariableTypeInfo();
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(), vbi.itemFamilyName(), vbi.itemGroupName(), vbi.meshName(), vti, sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
VariableRef*
VariableRefArrayT<DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
VariableRefArrayT<DataType>::
VariableRefArrayT(const VariableBuildInfo& vbi)
: VariableRef(vbi)
, m_private_part(PrivatePartType::getReference(vbi, _internalVariableInfo(vbi)))
{
  this->_internalInit(m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
VariableRefArrayT<DataType>::
VariableRefArrayT(const VariableRefArrayT<DataType>& rhs)
: VariableRef(rhs)
, ArrayView<DataType>(rhs)
, m_private_part(rhs.m_private_part)
{
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
VariableRefArrayT<DataType>::
VariableRefArrayT(IVariable* var)
: VariableRef(var)
, m_private_part(PrivatePartType::getReference(var))
{
  this->_internalInit(this->m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
void VariableRefArrayT<DataType>::
refersTo(const VariableRefArrayT<DataType>& rhs)
{
  VariableRef::operator=(rhs);
  m_private_part = rhs.m_private_part;
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Libère la mémoire allouée.
 */
template <typename DataType>
VariableRefArrayT<DataType>::
~VariableRefArrayT()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void VariableRefArrayT<DataType>::
resize(Integer s)
{
  m_private_part->resize(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void VariableRefArrayT<DataType>::
resizeWithReserve(Integer s, Integer nb_additional)
{
  m_private_part->resizeWithReserve(s, nb_additional);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void VariableRefArrayT<DataType>::
updateFromInternal()
{
  ArrayBase::setArray(m_private_part->valueView());
  BaseClass::updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
typename VariableRefArrayT<DataType>::LockType
VariableRefArrayT<DataType>::
lock()
{
  return LockType(m_private_part->trueData()->_internal()->_internalDeprecatedValue(), m_private_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
typename VariableRefArrayT<T>::ContainerType&
VariableRefArrayT<T>::
internalContainer()
{
  return _internalTrueData()->_internalDeprecatedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> IArrayDataInternalT<T>*
VariableRefArrayT<T>::
_internalTrueData()
{
  if (!(property() & IVariable::PPrivate))
    ARCANE_FATAL("variable '{0}': getting internal data container is only valid on private variable", name());
  return m_private_part->trueData()->_internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class VariableRefArrayT<Byte>;
template class VariableRefArrayT<Real>;
template class VariableRefArrayT<Int16>;
template class VariableRefArrayT<Int32>;
template class VariableRefArrayT<Int64>;
template class VariableRefArrayT<String>;
template class VariableRefArrayT<Real2>;
template class VariableRefArrayT<Real3>;
template class VariableRefArrayT<Real2x2>;
template class VariableRefArrayT<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
