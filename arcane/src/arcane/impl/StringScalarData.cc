// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringScalarData.cc                                         (C) 2000-2020 */
/*                                                                           */
/* Donnée scalaire de type 'String'.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/StringScalarData.h"

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"

#include "arcane/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringScalarData::
StringScalarData(const DataStorageBuildInfo& dsbi)
: m_trace(dsbi.traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringScalarData::
staticStorageTypeInfo()
{
  eBasicDataType bdt = eBasicDataType::Byte;
  Int32 nb_basic_type = 0;
  Int32 dimension = 1;
  Int32 multi_tag = 1;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringScalarData::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> StringScalarData::
createSerializedDataRef(bool use_basic_type) const
{
  return makeRef(const_cast<ISerializedData*>(createSerializedData(use_basic_type)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISerializedData* StringScalarData::
createSerializedData(bool use_basic_type) const
{
  ARCANE_UNUSED(use_basic_type);

  ByteConstArrayView local_values(m_value.utf8());
  Int64 nb_element = 1;
  Int64 len = local_values.size();
  Int64 nb_base_element = len;
  Int64UniqueArray extents;
  extents.add(nb_element);
  Span<const Byte> base_values = local_values;
  ISerializedData* sd = new SerializedData(DT_Byte, base_values.size(), 1, nb_element,
                                           nb_base_element, false, extents);
  sd->setBytes(base_values);
  //m_trace->info() << " WRITE STRING " << m_value << " len=" << len;
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  sdata->allocateMemory(sdata->memorySize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
assignSerializedData(const ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  Span<const Byte> byte_values = sdata->bytes();
  Int64 len = sdata->nbBaseElement();
  //m_trace->info() << " ASSIGN STRING n=" << len
  //                << " ptr=" << (void*)byte_values.begin();
  if (len != 0)
    m_value = String(byte_values);
  else
    m_value = String();
  //m_trace->info() << " READ STRING = " << m_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
serialize(ISerializer* sbuf, IDataOperation* operation)
{
  ARCANE_UNUSED(operation);

  ISerializer::eMode mode = sbuf->mode();
  if (mode == ISerializer::ModeReserve) {
    sbuf->reserve(m_value);
  }
  else if (mode == ISerializer::ModePut) {
    sbuf->put(m_value);
  }
  else if (mode == ISerializer::ModeGet) {
    switch (sbuf->readMode()) {
    case ISerializer::ReadReplace:
      sbuf->get(m_value);
      break;
    case ISerializer::ReadAdd:
      throw NotSupportedException(A_FUNCINFO, "option 'ReadAdd'");
    }
  }
  else
    throw NotSupportedException(A_FUNCINFO, "Invalid mode");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation)
{
  ARCANE_UNUSED(sbuf);
  ARCANE_UNUSED(ids);
  ARCANE_UNUSED(operation);
  // Rien à faire sur ce type de sérialisation.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
setName(const String& name)
{
  ARCANE_UNUSED(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
computeHash(IHashAlgorithm* algo, ByteArray& output) const
{
  ByteConstArrayView input = m_value.utf8();
  algo->computeHash(input, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
copy(const IData* data)
{
  const DataInterfaceType* true_data = dynamic_cast<const DataInterfaceType*>(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO, "Can not cast 'IData' to 'IScalarDataT'");
  m_value = true_data->value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
swapValues(IData* data)
{
  DataInterfaceType* true_data = dynamic_cast<DataInterfaceType*>(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO, "Can not cast 'IData' to 'IScalarDataT'");
  std::swap(m_value, true_data->value());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
visitScalar(IScalarDataVisitor* visitor)
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
visitArray(IArrayDataVisitor*)
{
  throw NotSupportedException(A_FUNCINFO, "Can not visit array data with scalar data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
visitArray2(IArray2DataVisitor*)
{
  throw NotSupportedException(A_FUNCINFO, "Can not visit array2 data with scalar data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringScalarData::
visitMultiArray2(IMultiArray2DataVisitor*)
{
  throw NotSupportedException(A_FUNCINFO, "Can not visit multiarray2 data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerStringScalarDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<StringScalarData>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
