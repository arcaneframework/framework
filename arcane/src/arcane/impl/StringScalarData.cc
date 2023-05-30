// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringScalarData.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Donnée scalaire de type 'String'.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayShape.h"

#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/datatype/DataStorageTypeInfo.h"
#include "arcane/core/datatype/DataStorageBuildInfo.h"
#include "arcane/core/datatype/DataTypeTraits.h"

#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"

#include "arcane/core/ISerializer.h"
#include "arcane/core/IDataVisitor.h"

#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Donnée scalaire d'une chaîne de caractères unicode.
 */
class StringScalarData
: public ReferenceCounterImpl
, public IScalarDataT<String>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  typedef String DataType;
  typedef StringScalarData ThatClass;
  typedef IScalarDataT<String> DataInterfaceType;

 public:
  explicit StringScalarData(ITraceMng* trace)
  : m_trace(trace) {}
  explicit StringScalarData(const DataStorageBuildInfo& dsbi);
  StringScalarData(const StringScalarData& rhs)
  : m_value(rhs.m_value)
  , m_trace(rhs.m_trace)
  , m_allocation_info(rhs.m_allocation_info)
  {}

 public:

  Integer dimension() const override { return 0; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  DataType& value() override { return m_value; }
  const DataType& value() const override { return m_value; }
  void resize(Integer) override {}
  IData* clone() override { return cloneTrue(); }
  IData* cloneEmpty() override { return cloneTrueEmpty(); };
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
  StringScalarData* cloneTrue() override { return new ThatClass(*this); }
  StringScalarData* cloneTrueEmpty() override { return new ThatClass(m_trace); }
  Ref<DataInterfaceType> cloneTrueRef() override { DataInterfaceType* d = new ThatClass(*this); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { DataInterfaceType* d = new ThatClass(m_trace); return makeRef(d); }
  void fillDefault() override
  {
    m_value = String();
  }
  void setName(const String& name) override;
  Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const override;
  void allocateBufferForSerializedData(ISerializedData* sdata) override;
  void assignSerializedData(const ISerializedData* sdata) override;
  void copy(const IData* data) override;
  void swapValues(IData* data) override;
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
  ArrayShape shape() const override { return {}; }
  void setShape(const ArrayShape&) override { }
  void setAllocationInfo(const DataAllocationInfo& v) override { m_allocation_info = v; }
  DataAllocationInfo allocationInfo() const override { return m_allocation_info; }
  void visit(IScalarDataVisitor* visitor) override
  {
    visitor->applyVisitor(this);
  }
  void visit(IDataVisitor* visitor) override
  {
    visitor->applyDataVisitor(this);
  }
  void visitScalar(IScalarDataVisitor* visitor) override;
  void visitArray(IArrayDataVisitor* visitor) override;
  void visitArray2(IArray2DataVisitor* visitor) override;
  IDataInternal* _commonInternal() override { return &m_internal; }

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  DataType m_value; //!< Donnée
  ITraceMng* m_trace;
  NullDataInternal m_internal;
  DataAllocationInfo m_allocation_info;
};

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
  ARCANE_UNUSED(use_basic_type);

  ByteConstArrayView local_values(m_value.utf8());
  Int64 nb_element = 1;
  Int64 len = local_values.size();
  Int64 nb_base_element = len;
  Int64UniqueArray extents;
  extents.add(nb_element);
  Span<const Byte> base_values = local_values;
  auto sd = arcaneCreateSerializedDataRef(DT_Byte, base_values.size(), 1, nb_element,
                                          nb_base_element, false, extents);
  sd->setConstBytes(base_values);
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

  Span<const Byte> byte_values = sdata->constBytes();
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
  algo->computeHash64(input, output);
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
