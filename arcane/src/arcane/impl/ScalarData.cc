// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScalarData.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Donnée de type scalaire.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IDataFactory.h"
#include "arcane/core/IData.h"
#include "arcane/core/IDataVisitor.h"

#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/datatype/DataStorageTypeInfo.h"
#include "arcane/core/datatype/DataStorageBuildInfo.h"
#include "arcane/core/datatype/DataTypeTraits.h"

#include "arcane/impl/ScalarData.h"
#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"

#include "arcane/core/ISerializer.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Donnée scalaire d'un type \a T
 */
template <class DataType>
class ScalarDataT
: public ReferenceCounterImpl
, public IScalarDataT<DataType>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  typedef ScalarDataT<DataType> ThatClass;
  typedef IScalarDataT<DataType> DataInterfaceType;

 public:

  class Internal
  : public IDataInternal
  {
   public:

    explicit Internal(ScalarDataT<DataType>* p) : m_p(p){}

   public:

    void computeHash(DataHashInfo& hash_info) override
    {
      m_p->computeHash(hash_info);
    }

   private:

    ScalarDataT<DataType>* m_p = nullptr;
  };

 public:

  explicit ScalarDataT(ITraceMng* trace)
  : m_value(DataTypeTraitsT<DataType>::defaultValue())
  , m_trace(trace)
  , m_internal(this)
  {}
  explicit ScalarDataT(const DataStorageBuildInfo& dsbi);
  ScalarDataT(const ScalarDataT<DataType>& rhs)
  : m_value(rhs.m_value)
  , m_trace(rhs.m_trace)
  , m_internal(this)
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
  IData* clone() override { return _cloneTrue(); }
  IData* cloneEmpty() override { return _cloneTrueEmpty(); };
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
  DataInterfaceType* cloneTrue() override { return _cloneTrue(); }
  DataInterfaceType* cloneTrueEmpty() override { return _cloneTrueEmpty(); }
  Ref<DataInterfaceType> cloneTrueRef() override { auto* d = _cloneTrue(); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { auto* d = _cloneTrueEmpty(); return makeRef(d); }
  void fillDefault() override;
  void setName(const String& name) override;
  Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const override;
  void allocateBufferForSerializedData(ISerializedData* sdata) override;
  void assignSerializedData(const ISerializedData* sdata) override;
  void copy(const IData* data) override;
  void swapValues(IData* data) override;
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
  void computeHash(DataHashInfo& hash_info) const;
  ArrayShape shape() const override;
  void setShape(const ArrayShape&) override;
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
  Internal m_internal;
  DataAllocationInfo m_allocation_info;

 private:

  DataInterfaceType* _cloneTrue() const { return new ThatClass(*this); }
  DataInterfaceType* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ScalarDataT<DataType>::
ScalarDataT(const DataStorageBuildInfo& dsbi)
: m_value(DataTypeTraitsT<DataType>::defaultValue())
, m_trace(dsbi.traceMng())
, m_internal(this)
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
  eBasicDataType data_type = DataTypeTraitsT<BasicType>::basicDataType();

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
  algo->computeHash64(input, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
computeHash(DataHashInfo& hash_info) const
{
  hash_info.setVersion(2);
  Span<const DataType> value_as_span(&m_value,1);
  hash_info.context()->updateHash(asBytes(value_as_span));
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
ArrayShape ScalarDataT<DataType>::
shape() const
{
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
void ScalarDataT<DataType>::
setShape(const ArrayShape&)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerScalarDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<ScalarDataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<BFloat16>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Float16>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Float32>>::registerDataFactory(dfm);
  DataStorageFactory<ScalarDataT<Int8>>::registerDataFactory(dfm);
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

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(ScalarDataT);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
