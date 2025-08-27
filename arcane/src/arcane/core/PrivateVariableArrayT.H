// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableArrayT.H                                     (C) 2000-2020 */
/*                                                                           */
/* Factorisation de variable array du maillage.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/PrivateVariableArray.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Variable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
PrivateVariableArrayT<DataType>::
PrivateVariableArrayT(const VariableBuildInfo& vb, const VariableInfo& vi)
: MeshVariableRef(vb)
, m_private_part(PrivatePartType::getReference(vb,vi))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
PrivateVariableArrayT<DataType>::
PrivateVariableArrayT(IVariable* var)
: MeshVariableRef(var)
, m_private_part(PrivatePartType::getReference(var))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
PrivateVariableArrayT<DataType>::
PrivateVariableArrayT(const PrivateVariableArrayT& rhs)
: MeshVariableRef(rhs)
, m_private_part(rhs.m_private_part)
{
  if (m_private_part)
    updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void 
PrivateVariableArrayT<DataType>::
operator=(const PrivateVariableArrayT& rhs)
{
  if (this != &rhs) {
    VariableRef::operator=(rhs);
    m_private_part = rhs.m_private_part;
    if (m_private_part)
      updateFromInternal();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
PrivateVariableArrayT<DataType>::
updateFromInternal()
{
  ARCANE_CHECK_POINTER(m_private_part);
  m_view = m_private_part->valueView();
  MeshVariableRef::updateFromInternal();
  _executeUpdateFunctors();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
PrivateVariableArrayT<DataType>::
resize(Integer dim2_size)
{
  ARCANE_CHECK_POINTER(m_private_part);
  m_private_part->directResize(m_private_part->valueView().dim1Size(),dim2_size);
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
PrivateVariableArrayT<DataType>::
resizeAndReshape(const ArrayShape& shape)
{
  m_private_part->directResizeAndReshape(shape);
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> ItemGroup
PrivateVariableArrayT<DataType>::
itemGroup() const
{
  return m_private_part->itemGroup();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
