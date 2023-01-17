﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractDataVisitor.cc                                      (C) 2000-2016 */
/*                                                                           */
/* Visiteur abstrait pour une donnée.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/AbstractDataVisitor.h"
#include "arcane/IData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractDataVisitor::
applyDataVisitor(IScalarData* data)
{
  data->visitScalar(this);
}

void AbstractDataVisitor::
applyDataVisitor(IArrayData* data)
{
  data->visitArray(this);
}

void AbstractDataVisitor::
applyDataVisitor(IArray2Data* data)
{
  data->visitArray2(this);
}

void AbstractDataVisitor::
applyDataVisitor(IMultiArray2Data* data)
{
  data->visitMultiArray2(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractScalarDataVisitor::
_throwException(eDataType dt)
{
  String s = String::format("scalar visitor not implemented for data type '{0}'",
                            dataTypeName(dt));
  throw NotImplementedException(A_FUNCINFO,s);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Byte>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Byte);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Real>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Int16>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int16);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Int32>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int32);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Int64>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int64);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Real2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Real3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Real2x2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2x2);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<Real3x3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3x3);
}

void AbstractScalarDataVisitor::
applyVisitor(IScalarDataT<String>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_String);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractArrayDataVisitor::
_throwException(eDataType dt)
{
  String s = String::format("array visitor not implemented for data type '{0}'",
                            dataTypeName(dt));
  throw NotImplementedException(A_FUNCINFO,s);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Byte>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Byte);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Real>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Int16>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int16);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Int32>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int32);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Int64>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int64);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Real2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Real3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Real2x2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2x2);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<Real3x3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3x3);
}

void AbstractArrayDataVisitor::
applyVisitor(IArrayDataT<String>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_String);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractArray2DataVisitor::
_throwException(eDataType dt)
{
  String s = String::format("array2 visitor not implemented for data type '{0}'",
                            dataTypeName(dt));
  throw NotImplementedException(A_FUNCINFO,s);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Byte>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Byte);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Real>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Int16>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int16);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Int32>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int32);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Int64>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int64);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Real2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Real3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Real2x2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2x2);
}

void AbstractArray2DataVisitor::
applyVisitor(IArray2DataT<Real3x3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3x3);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractMultiArray2DataVisitor::
_throwException(eDataType dt)
{
  String s = String::format("multiarray2 visitor not implemented for data type '{0}'",
                            dataTypeName(dt));
  throw NotImplementedException(A_FUNCINFO,s);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Byte>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Byte);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Real>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Int16>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int16);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Int32>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int32);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Int64>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Int64);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Real2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Real3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Real2x2>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real2x2);
}

void AbstractMultiArray2DataVisitor::
applyVisitor(IMultiArray2DataT<Real3x3>* data)
{
  ARCANE_UNUSED(data);
  _throwException(DT_Real3x3);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
