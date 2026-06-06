// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Reference to a variable on a mesh material.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MeshEnvironmentVariableRef.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NumericTypes.h"

#include "arcane/MeshVariableScalarRef.h"
#include "arcane/ArcaneException.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const VariableBuildInfo& vb)
: CellEnvironmentVariableScalarRef(MaterialVariableBuildInfo(nullptr, vb))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(vb, MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> CellEnvironmentVariableScalarRef<DataType>::
CellEnvironmentVariableScalarRef(const CellEnvironmentVariableScalarRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
, m_value(nullptr)
{
  // The reference counter must be manually incremented because normally
  // this is done in getReference(), but here it is not called
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
_init()
{
  if (m_private_part) {
    this->_setContainerView();
    _internalInit(m_private_part->toMeshMaterialVariable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
refersTo(const CellEnvironmentVariableScalarRef<DataType>& rhs)
{
  if (rhs.m_private_part == m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();

  m_private_part = rhs.m_private_part;
  m_value = nullptr;
  m_container_value = {};

  // The reference counter must be manually incremented because normally
  // this is done in getReference(), but here it is not called
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
updateFromInternal()
{
  _setContainerView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> DataType
CellEnvironmentVariableScalarRef<DataType>::
envValue(AllEnvCell c, Int32 env_id) const
{
  ENUMERATE_CELL_ENVCELL (ienvcell, c) {
    EnvCell ec = *ienvcell;
    Int32 eid = ec.environmentId();
    if (eid == env_id)
      return this->operator[](ienvcell);
  }
  return DataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills the partial and global values of the variable with the value \a value
 */
template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
fill(const DataType& value)
{
  globalVariable().fill(value);
  fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills the partial values of the variable with the value \a value
 */
template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
fillPartialValues(const DataType& value)
{
  m_private_part->fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> MeshVariableScalarRefT<Cell, DataType>&
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

template <typename DataType> const MeshVariableScalarRefT<Cell, DataType>&
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

template <typename DataType> void
CellEnvironmentVariableScalarRef<DataType>::
_setContainerView()
{
  if (m_private_part) {
    m_container_value = m_private_part->_internalFullValuesView();
    m_value = m_container_value.data();
  }
  else {
    m_container_value = {};
    m_value = nullptr;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const VariableBuildInfo& vb)
: CellEnvironmentVariableArrayRef(MaterialVariableBuildInfo(nullptr, vb))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(vb, MatVarSpace::Environment))
, m_value(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> CellEnvironmentVariableArrayRef<DataType>::
CellEnvironmentVariableArrayRef(const CellEnvironmentVariableArrayRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
, m_value(nullptr)
{
  // The reference counter must be manually incremented because normally
  // this is done in getReference(), but here it is not called
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
_init()
{
  if (m_private_part) {
    _setContainerView();
    _internalInit(m_private_part->toMeshMaterialVariable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
refersTo(const CellEnvironmentVariableArrayRef<DataType>& rhs)
{
  if (rhs.m_private_part == m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();

  m_private_part = rhs.m_private_part;
  m_value = nullptr;
  m_container_value = {};

  // The reference counter must be manually incremented because normally
  // this is done in getReference(), but here it is not called
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: merge with the scalar version
template <typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
updateFromInternal()
{
  _setContainerView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> MeshVariableArrayRefT<Cell, DataType>&
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

template <typename DataType> const MeshVariableArrayRefT<Cell, DataType>&
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

template <typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
resize(Integer dim2_size)
{
  m_private_part->resize(dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
CellEnvironmentVariableArrayRef<DataType>::
_setContainerView()
{
  if (m_private_part) {
    m_container_value = m_private_part->_internalFullValuesView();
    m_value = m_container_value.data();
  }
  else {
    m_container_value = {};
    m_value = nullptr;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ARCANE_TEMPLATE_EXPORT CellEnvironmentVariableScalarRef<type>; \
  template class ARCANE_TEMPLATE_EXPORT CellEnvironmentVariableArrayRef<type>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int8);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(BFloat16);
ARCANE_INSTANTIATE_MAT(Float16);
ARCANE_INSTANTIATE_MAT(Float32);
ARCANE_INSTANTIATE_MAT(Real);
ARCANE_INSTANTIATE_MAT(Real2);
ARCANE_INSTANTIATE_MAT(Real3);
ARCANE_INSTANTIATE_MAT(Real2x2);
ARCANE_INSTANTIATE_MAT(Real3x3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
