// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Référence à une variable sur un matériau du maillage.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/MeshVariableScalarRef.h"
#include "arcane/VariableBuildInfo.h"

#include "arcane/ArcaneException.h"

#include "arcane/materials/MeshEnvironmentVariableRef.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const VariableBuildInfo& vb)
: m_private_part(PrivatePartType::getReference(vb,nullptr,MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::getReference(vb,MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>  CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const CellEnvironmentVariableScalarRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
, m_value(nullptr)
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
_init()
{
  if (m_private_part){
    m_value = m_private_part->views();
    _internalInit(m_private_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
refersTo(const CellEnvironmentVariableScalarRef<DataType>& rhs)
{
  if (rhs.m_private_part==m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();
  m_private_part = rhs.m_private_part;
  m_value = nullptr;

  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
updateFromInternal()
{
  m_value = m_private_part->views();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataType
CellEnvironmentVariableScalarRef<DataType>::
envValue(AllEnvCell c,Int32 env_id) const
{
  ENUMERATE_CELL_ENVCELL(ienvcell,c){
    EnvCell ec = *ienvcell;
    Int32 eid = ec.environmentId();
    if (eid==env_id)
      return this->operator[](ienvcell);
  }
  return DataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs partielles et globales de la variable avec la valeur \a value
 */
template<typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
fill(const DataType& value)
{
  globalVariable().fill(value);
  fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs partielles de la variable avec la valeur \a value
 */
template<typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
fillPartialValues(const DataType& value)
{
  m_private_part->fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> MeshVariableScalarRefT<Cell,DataType>&
CellEnvironmentVariableScalarRef<DataType>::
globalVariable()
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> const MeshVariableScalarRefT<Cell,DataType>&
CellEnvironmentVariableScalarRef<DataType>::
globalVariable() const
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const VariableBuildInfo& vb)
: m_private_part(PrivatePartType::getReference(vb,nullptr,MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::getReference(vb,MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType>  CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const CellEnvironmentVariableArrayRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
, m_value(nullptr)
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
_init()
{
  if (m_private_part){
    m_value = m_private_part->views();
    _internalInit(m_private_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
refersTo(const CellEnvironmentVariableArrayRef<DataType>& rhs)
{
  if (rhs.m_private_part==m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();
  m_private_part = rhs.m_private_part;
  m_value = nullptr;

  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
updateFromInternal()
{
  m_value = m_private_part->views();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> MeshVariableArrayRefT<Cell,DataType>&
CellEnvironmentVariableArrayRef<DataType>::
globalVariable()
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> const MeshVariableArrayRefT<Cell,DataType>&
CellEnvironmentVariableArrayRef<DataType>::
globalVariable() const
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
resize(Integer dim2_size)
{
  m_private_part->resize(dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class CellEnvironmentVariableScalarRef<Byte>;
template class CellEnvironmentVariableScalarRef<Real>;
template class CellEnvironmentVariableScalarRef<Int16>;
template class CellEnvironmentVariableScalarRef<Int32>;
template class CellEnvironmentVariableScalarRef<Int64>;
template class CellEnvironmentVariableScalarRef<Real2>;
template class CellEnvironmentVariableScalarRef<Real3>;
template class CellEnvironmentVariableScalarRef<Real2x2>;
template class CellEnvironmentVariableScalarRef<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class CellEnvironmentVariableArrayRef<Byte>;
template class CellEnvironmentVariableArrayRef<Real>;
template class CellEnvironmentVariableArrayRef<Int16>;
template class CellEnvironmentVariableArrayRef<Int32>;
template class CellEnvironmentVariableArrayRef<Int64>;
template class CellEnvironmentVariableArrayRef<Real2>;
template class CellEnvironmentVariableArrayRef<Real3>;
template class CellEnvironmentVariableArrayRef<Real2x2>;
template class CellEnvironmentVariableArrayRef<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
