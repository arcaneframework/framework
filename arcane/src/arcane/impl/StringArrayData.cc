// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringArrayData.cc                                          (C) 2000-2023 */
/*                                                                           */
/* Donnée de type 'UniqueArray<String>'.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IHashAlgorithm.h"
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
 * \brief Donnée tableau d'une chaîne de caractères unicode (spécialisation)
 */
class StringArrayData
: public ReferenceCounterImpl
, public IArrayDataT<String>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
  class Impl;
  friend class Impl;

public:

  typedef String DataType;
  typedef StringArrayData ThatClass;
  typedef IArrayDataT<String> DataInterfaceType;

 public:
  explicit StringArrayData(ITraceMng* trace);
  explicit StringArrayData(const DataStorageBuildInfo& dsbi);
  StringArrayData(const StringArrayData& rhs);
  ~StringArrayData() override;
 public:
  Integer dimension() const override { return 1; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  Array<DataType>& value() override { return m_value; }
  const Array<DataType>& value() const override { return m_value; }
  ConstArrayView<DataType> view() const override { return m_value; }
  ArrayView<DataType> view() override { return m_value; }
  void resize(Integer new_size) override { m_value.resize(new_size); }
  IData* clone() override { return cloneTrue(); }
  IData* cloneEmpty() override { return cloneTrueEmpty(); };
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataInterfaceType* cloneTrue() override { return _cloneTrue(); }
  DataInterfaceType* cloneTrueEmpty() override { return _cloneTrueEmpty(); }
  Ref<DataInterfaceType> cloneTrueRef() override { auto* d = _cloneTrue(); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { auto* d = _cloneTrueEmpty(); return makeRef(d); }
  DataStorageTypeInfo storageTypeInfo() const override;
  void fillDefault() override { m_value.fill(String()); }
  void setName(const String& name) override;
  Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const override;
  void allocateBufferForSerializedData(ISerializedData* sdata) override;
  void assignSerializedData(const ISerializedData* sdata) override;
  void copy(const IData* data) override;
  void swapValues(IData* data) override;
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
  void computeHash(DataHashInfo& hash_info) const;
  ArrayShape shape() const override { return {}; }
  void setShape(const ArrayShape&) override {}
  void setAllocationInfo(const DataAllocationInfo& v) override { m_allocation_info = v; }
  DataAllocationInfo allocationInfo() const override { return m_allocation_info; }
  void visit(IArrayDataVisitor* visitor) override
  {
    visitor->applyVisitor(this);
  }
  void visit(IDataVisitor* visitor) override
  {
    visitor->applyDataVisitor((IArrayData*)this);
  }
  void visitScalar(IScalarDataVisitor* visitor) override;
  void visitArray(IArrayDataVisitor* visitor) override;
  void visitArray2(IArray2DataVisitor* visitor) override;

 public:

  IArrayDataInternalT<DataType>* _internal() override { return m_internal; }
  IDataInternal* _commonInternal() override { return m_internal; }

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  UniqueArray<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;
  IArrayDataInternalT<String>* m_internal;
  DataAllocationInfo m_allocation_info;

 private:

  ThatClass* _cloneTrue() const { return new ThatClass(*this); }
  ThatClass* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: à fusionner avec l'implémentation commune dans ArrayData.
class StringArrayData::Impl
: public IArrayDataInternalT<String>
{
 public:

  using String = DataType;

  explicit Impl(StringArrayData* p) : m_p(p){}

 public:

  void reserve(Integer new_capacity) override { m_p->m_value.reserve(new_capacity); }
  Array<DataType>& _internalDeprecatedValue() override { return m_p->m_value; }
  Integer capacity() const override { return m_p->m_value.capacity(); }
  void shrink() const override { m_p->m_value.shrink(); }
  void resize(Integer new_size) override { m_p->m_value.resize(new_size);}
  void dispose() override { m_p->m_value.dispose(); }
  void computeHash(DataHashInfo& hash_info) override
  {
    m_p->computeHash(hash_info);
  }

 private:

  StringArrayData* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringArrayData::
StringArrayData(const DataStorageBuildInfo& dsbi)
: m_trace(dsbi.traceMng())
, m_internal(new Impl(this))
{
}

StringArrayData::
StringArrayData(ITraceMng* trace)
: m_trace(trace)
, m_internal(new Impl(this))
{}

StringArrayData::
StringArrayData(const StringArrayData& rhs)
: m_value(rhs.m_value)
, m_trace(rhs.m_trace)
, m_internal(new Impl(this))
, m_allocation_info(rhs.m_allocation_info)
{}

StringArrayData::
~StringArrayData()
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringArrayData::
staticStorageTypeInfo()
{
  eBasicDataType bdt = eBasicDataType::Byte;
  Int32 nb_basic_type = 0;
  Int32 dimension = 2;
  Int32 multi_tag = 1;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringArrayData::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> StringArrayData::
createSerializedDataRef(bool use_basic_type) const
{
  ARCANE_UNUSED(use_basic_type);

  // Positionne les dimensions et calcule la taille nécessaire pour sérialiser
  // les valeurs
  Int64 needed_memory = 0;
  Int64 nb_element = m_value.largeSize();
  Int64UniqueArray dimensions(nb_element);
  for (Integer i = 0; i < nb_element; ++i) {
    Span<const Byte> str(m_value[i].bytes());
    Int64 len = str.size();
    needed_memory += len;
    dimensions[i] = len;
  }
  Int64 nb_base_element = needed_memory;
  auto sd = arcaneCreateSerializedDataRef(DT_Byte, needed_memory, 2, nb_element,
                                          nb_base_element, true, dimensions);
  sd->allocateMemory(needed_memory);

  // Recopie les valeurs dans le tableau alloué
  Span<Byte> svalues = sd->writableBytes();
  {
    Int64 index = 0;
    for (Integer i = 0; i < nb_element; ++i) {
      Span<const Byte> str(m_value[i].bytes());
      Int64 len = str.size();
      // TODO: utiliser directement une méthode de copie.
      for (Int64 z = 0; z < len; ++z)
        svalues[index + z] = str[z];
      index += len;
    }
  }
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  sdata->allocateMemory(sdata->memorySize());
  //m_trace->info() << " ALLOC ARRAY STRING ptr=" << (void*)byte_values.begin();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
assignSerializedData(const ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  Span<const Byte> byte_values = sdata->constBytes();
  //m_trace->info() << " ASSIGN ARRAY STRING ptr=" << (void*)byte_values.begin();
  Int64ConstArrayView dimensions = sdata->extents();
  Integer nb_element = dimensions.size();
  m_value.resize(nb_element);
  Int64 index = 0;
  for (Integer i = 0; i < nb_element; ++i) {
    Int64 len = dimensions[i];
    Span<const Byte> v(&byte_values[index], len);
    m_value[i] = String(v);
    index += len;
    //m_trace->info() << " READ STRING i=" << i << " v=" << m_value[i] << " len=" << len << " index=" << index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
serialize(ISerializer* sbuf, IDataOperation* operation)
{
  // TODO: tester cette méthode.
  ARCANE_UNUSED(operation);

  ISerializer::eMode mode = sbuf->mode();
  if (mode == ISerializer::ModeReserve) {
    Integer size = m_value.size();
    sbuf->reserveInteger(1);
    for (Integer z = 0; z < size; ++z)
      sbuf->reserve(m_value[z]);
  }
  else if (mode == ISerializer::ModePut) {
    Integer size = m_value.size();
    sbuf->putInteger(size);
    for (Integer z = 0; z < size; ++z)
      sbuf->put(m_value[z]);
  }
  else if (mode == ISerializer::ModeGet) {
    switch (sbuf->readMode()) {
    case ISerializer::ReadReplace: {
      Integer size = sbuf->getInteger();
      m_value.resize(size);
      for (Integer z = 0; z < size; ++z)
        sbuf->get(m_value[z]);
    } break;
    case ISerializer::ReadAdd:
      ARCANE_THROW(NotSupportedException, "option 'ReadAdd'");
    }
  }
  else
    ARCANE_THROW(NotSupportedException, "Invalid mode");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation)
{
  ARCANE_UNUSED(sbuf);
  ARCANE_UNUSED(ids);
  ARCANE_UNUSED(operation);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
setName(const String& name)
{
  ARCANE_UNUSED(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
computeHash(IHashAlgorithm* algo, ByteArray& output) const
{
  // Pour l'instant, il faut passer par une sérialisation.
  // TODO supprimer la sérialisation inutile.
  Ref<ISerializedData> s = createSerializedDataRef(true);
  s->computeHash(algo, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
computeHash(DataHashInfo& hash_info) const
{
  hash_info.setVersion(2);
  IHashAlgorithmContext* context = hash_info.context();
  for( const String& x : m_value )
    context->updateHash(asBytes(x.bytes()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
copy(const IData* data)
{
  const auto* true_data = dynamic_cast<const DataInterfaceType*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'StringArrayData'");
  m_value.copy(true_data->view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'StringArrayData'");
  m_value.swap(true_data->m_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitScalar(IScalarDataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit scalar data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitArray(IArrayDataVisitor* visitor)
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitArray2(IArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit array2 data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerStringArrayDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<StringArrayData>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
