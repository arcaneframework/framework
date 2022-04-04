﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScalarData.cc                                               (C) 2000-2020 */
/*                                                                           */
/* Donnée de type scalaire.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/IDataFactory.h"
#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/impl/ScalarData.h"
#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"
#include "arcane/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ScalarDataT<DataType>::
ScalarDataT(const DataStorageBuildInfo& dsbi)
: m_value(DataTypeTraitsT<DataType>::defaultValue())
, m_trace(dsbi.traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
DataStorageTypeInfo ScalarDataT<DataType>::
staticStorageTypeInfo()
{
  typedef DataTypeTraitsT<DataType> TraitsType;
  eBasicDataType bdt = TraitsType::basicDataType();
  Int32 nb_basic_type = TraitsType::nbBasicType();
  Int32 dimension = 0;
  Int32 multi_tag = 0;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
DataStorageTypeInfo ScalarDataT<DataType>::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
Ref<ISerializedData> ScalarDataT<DataType>::
createSerializedDataRef(bool use_basic_type) const
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  Integer nb_count = 1;
  eDataType data_type = dataType();
  Integer type_size = sizeof(DataType);

  if (use_basic_type) {
    nb_count = DataTypeTraitsT<DataType>::nbBasicType();
    data_type = DataTypeTraitsT<BasicType>::type();
    type_size = sizeof(BasicType);
  }

  Integer nb_element = 1;
  Integer nb_base_element = nb_element * nb_count;
  Integer full_size = nb_base_element * type_size;
  ByteConstArrayView base_values(full_size, reinterpret_cast<const Byte*>(&m_value));
  UniqueArray<Int64> dimensions;
  dimensions.add(nb_element);
  auto sd = arcaneCreateSerializedDataRef(data_type, base_values.size(), 0, nb_element,
                                          nb_base_element, false, dimensions);
  sd->setConstBytes(base_values);
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  eDataType data_type = sdata->baseDataType();
  eDataType base_data_type = DataTypeTraitsT<BasicType>::type();

  if (data_type != dataType() && data_type != base_data_type)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  Span<Byte> byte_values(reinterpret_cast<Byte*>(&m_value), sdata->memorySize());
  sdata->setWritableBytes(byte_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
assignSerializedData(const ISerializedData*)
{
  // Rien à faire car \a sdata pointe directement vers m_value
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
serialize(ISerializer* sbuf, IDataOperation*)
{
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();

  DataType ttmp = m_value;
  ArrayView<BasicType> vtmp(1 * nb_count, reinterpret_cast<BasicType*>(&ttmp));

  ISerializer::eMode mode = sbuf->mode();
  if (mode == ISerializer::ModeReserve)
    sbuf->reserveSpan(data_type, vtmp.size());
  else if (mode == ISerializer::ModePut)
    sbuf->putSpan(vtmp);
  else if (mode == ISerializer::ModeGet)
    sbuf->getSpan(vtmp);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
serialize(ISerializer*, Int32ConstArrayView, IDataOperation*)
{
  // Rien à faire.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
fillDefault()
{
  m_value = DataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
setName(const String&)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
computeHash(IHashAlgorithm* algo, ByteArray& output) const
{
  Integer type_size = sizeof(DataType);
  const Byte* ptr = reinterpret_cast<const Byte*>(&m_value);
  ByteConstArrayView input(type_size, ptr);
  algo->computeHash(input, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
copy(const IData* data)
{
  const DataInterfaceType* true_data = dynamic_cast<const DataInterfaceType*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'IScalarDataT'");
  m_value = true_data->value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
swapValues(IData* data)
{
  DataInterfaceType* true_data = dynamic_cast<DataInterfaceType*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'IScalarDataT'");
  std::swap(m_value, true_data->value());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
visitScalar(IScalarDataVisitor* visitor)
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
visitArray(IArrayDataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit array data with scalar data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
visitArray2(IArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit array2 data with scalar data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
visitMultiArray2(IMultiArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit multiarray2 data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerScalarDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<ScalarDataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Int16>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Int32>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Int64>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real2>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real3>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real2x2>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real3x3>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ScalarDataT<Byte>;
template class ScalarDataT<Real>;
template class ScalarDataT<Int16>;
template class ScalarDataT<Int32>;
template class ScalarDataT<Int64>;
template class ScalarDataT<Real2>;
template class ScalarDataT<Real3>;
template class ScalarDataT<Real2x2>;
template class ScalarDataT<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
