// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTypeDispatchingDataVisitor.cc                           (C) 2000-2016 */
/*                                                                           */
/* IDataVisitor dispatchant les opérations suivant le type de donnée.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/DataTypeDispatchingDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractDataTypeDispatchingDataVisitor::
AbstractDataTypeDispatchingDataVisitor(IDataTypeDataDispatcherT<Byte>* a_byte,
                                       IDataTypeDataDispatcherT<Real>* a_real,
                                       IDataTypeDataDispatcherT<Int16>* a_int16,
                                       IDataTypeDataDispatcherT<Int32>* a_int32,
                                       IDataTypeDataDispatcherT<Int64>* a_int64,
                                       IDataTypeDataDispatcherT<Real2>* a_real2,
                                       IDataTypeDataDispatcherT<Real3>* a_real3,
                                       IDataTypeDataDispatcherT<Real2x2>* a_real2x2,
                                       IDataTypeDataDispatcherT<Real3x3>* a_real3x3,
                                       IDataTypeDataDispatcherT<String>* a_string
                                       )
{
  m_byte = a_byte;
  m_real = a_real;
  m_int16 = a_int16;
  m_int32 = a_int32;
  m_int64 = a_int64;
  m_real2 = a_real2;
  m_real3 = a_real3;
  m_real2x2 = a_real2x2;
  m_real3x3 = a_real3x3;
  m_string = a_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractDataTypeDispatchingDataVisitor::
~AbstractDataTypeDispatchingDataVisitor()
{
  delete m_byte;
  delete m_real;
  delete m_int16;
  delete m_int32;
  delete m_int64;
  delete m_real2;
  delete m_real3;
  delete m_real2x2;
  delete m_real3x3;
  delete m_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Byte>* data)
{
  m_byte->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Real>* data)
{
  m_real->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Int32>* data)
{
  m_int32->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Int16>* data)
{
  m_int16->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Int64>* data)
{
  m_int64->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Real2>* data)
{
  m_real2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Real3>* data)
{
  m_real3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Real2x2>* data)
{
  m_real2x2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<Real3x3>* data)
{
  m_real3x3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IScalarDataT<String>* data)
{
  m_string->applyDispatch(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Byte>* data)
{
  m_byte->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Real>* data)
{
  m_real->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Int16>* data)
{
  m_int16->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Int32>* data)
{
  m_int32->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Int64>* data)
{
  m_int64->applyDispatch(data);
}
   
void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Real2>* data)
{
  m_real2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Real3>* data)
{
  m_real3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Real2x2>* data)
{
  m_real2x2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<Real3x3>* data)
{
  m_real3x3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArrayDataT<String>* data)
{
  m_string->applyDispatch(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Byte>* data)
{
  m_byte->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Real>* data)
{
  m_real->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Int16>* data)
{
  m_int16->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Int32>* data)
{
  m_int32->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Int64>* data)
{
  m_int64->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Real2>* data)
{
  m_real2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Real3>* data)
{
  m_real3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Real2x2>* data)
{
  m_real2x2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IArray2DataT<Real3x3>* data)
{
  m_real3x3->applyDispatch(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Byte>* data)
{
  m_byte->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Real>* data)
{
  m_real->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Int16>* data)
{
  m_int16->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Int32>* data)
{
  m_int32->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Int64>* data)
{
  m_int64->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Real2>* data)
{
  m_real2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Real3>* data)
{
  m_real3->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Real2x2>* data)
{
  m_real2x2->applyDispatch(data);
}

void AbstractDataTypeDispatchingDataVisitor::
applyVisitor(IMultiArray2DataT<Real3x3>* data)
{
  m_real3x3->applyDispatch(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
