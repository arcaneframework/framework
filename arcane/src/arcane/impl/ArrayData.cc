// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayData.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Donnée du type 'Array'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IndexOutOfRangeException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/datatype/DataStorageBuildInfo.h"
#include "arcane/core/datatype/IDataOperation.h"
#include "arcane/core/datatype/DataStorageTypeInfo.h"
#include "arcane/core/datatype/DataTypeTraits.h"

#include "arcane/core/ISerializer.h"
#include "arcane/core/IData.h"
#include "arcane/core/IDataVisitor.h"

#include "arcane/core/internal/IDataInternal.h"

#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace 
{
  const Int64 SERIALIZE_MAGIC_NUMBER = 0x456ff989;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Donnée tableau d'un type \a T
 */
template<class DataType>
class ArrayDataT
: public ReferenceCounterImpl
, public IArrayDataT<DataType>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
  class Impl;
  friend class Impl;

 public:

  typedef ArrayDataT<DataType> ThatClass;
  typedef IArrayDataT<DataType> DataInterfaceType;

 public:

  explicit ArrayDataT(ITraceMng* trace);
  explicit ArrayDataT(const DataStorageBuildInfo& dsbi);
  ArrayDataT(const ArrayDataT<DataType>& rhs);
  ~ArrayDataT() override;

 public:

  Integer dimension() const override { return 1; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf,IDataOperation* operation) override;
  void serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation) override;
  Array<DataType>& value() override { return m_value; }
  const Array<DataType>& value() const override { return m_value; }
  ConstArrayView<DataType> view() const override { return m_value; }
  ArrayView<DataType> view() override { return m_value; }
  void resize(Integer new_size) override { m_value.resize(new_size); }
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
  void computeHash(IHashAlgorithm* algo,ByteArray& output) const override;
  void computeHash(DataHashInfo& hash_info) const;
  ArrayShape shape() const override { return m_shape; }
  void setShape(const ArrayShape& new_shape) override { m_shape = new_shape; }
  void setAllocationInfo(const DataAllocationInfo& v) override;
  DataAllocationInfo allocationInfo() const override { return m_allocation_info; }
  void visit(IArrayDataVisitor* visitor) override
  {
    visitor->applyVisitor(this);
  }
  void visit(IDataVisitor* visitor) override
  {
    visitor->applyDataVisitor(this);
  }
  void visitScalar(IScalarDataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException,"Can not visit scalar data with array data");
  }
  void visitArray(IArrayDataVisitor* visitor) override
  {
    visitor->applyVisitor(this);
  }
  void visitArray2(IArray2DataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException,"Can not visit array2 data with array data");
  }

 public:

  void swapValuesDirect(ThatClass* true_data);

 public:

  IArrayDataInternalT<DataType>* _internal() override { return m_internal; }
  IDataInternal* _commonInternal() override { return m_internal; }

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 public:


 private:

  UniqueArray<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;
  IArrayDataInternalT<DataType>* m_internal;
  ArrayShape m_shape;
  DataAllocationInfo m_allocation_info;

 private:

  void _serialize(ISerializer* sbuf,Span<const Int32> ids,IDataOperation* operation);
  IArrayDataT<DataType>* _cloneTrue() const { return new ThatClass(*this); }
  IArrayDataT<DataType>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
  void _setShape();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class ArrayDataT<DataType>::Impl
: public IArrayDataInternalT<DataType>
, public INumericDataInternal
{
 public:

  explicit Impl(ArrayDataT<DataType>* p) : m_p(p){}

 public:

  void reserve(Integer new_capacity) override { m_p->m_value.reserve(new_capacity); }
  Array<DataType>& _internalDeprecatedValue() override { return m_p->m_value; }
  Integer capacity() const override { return m_p->m_value.capacity(); }
  void shrink() const override { m_p->m_value.shrink(); }
  void resize(Integer new_size) override { m_p->m_value.resize(new_size);}
  void dispose() override { m_p->m_value.dispose(); }
  bool compressAndClear(DataCompressionBuffer& buf) override
  {
    IDataCompressor* compressor = buf.m_compressor;
    if (!compressor)
      return false;
    Span<const DataType> values = m_p->m_value;
    Span<const std::byte> bytes = asBytes(values);
    compressor->compress(bytes,buf.m_buffer);
    buf.m_original_dim1_size = values.size();
    m_p->m_value.clear();
    m_p->m_value.shrink();
    return true;
  }
  bool decompressAndFill(DataCompressionBuffer& buf) override
  {
    IDataCompressor* compressor = buf.m_compressor;
    if (!compressor)
      return false;
    m_p->m_value.resize(buf.m_original_dim1_size);
    Span<DataType> values = m_p->m_value;
    compressor->decompress(buf.m_buffer,asWritableBytes(values));
    return true;
  }
  MutableMemoryView memoryView() override
  {
    return makeMutableMemoryView<DataType>(m_p->view());
  }
  INumericDataInternal* numericData() override { return this; }
  void computeHash(DataHashInfo& hash_info) override
  {
    m_p->computeHash(hash_info);
  }

 private:

  ArrayDataT<DataType>* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ArrayDataT<DataType>::
ArrayDataT(ITraceMng* trace)
: m_value(AlignedMemoryAllocator::Simd())
, m_trace(trace)
, m_internal(new Impl(this))
{
  _setShape();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ArrayDataT<DataType>::
ArrayDataT(const ArrayDataT<DataType>& rhs)
: m_value(AlignedMemoryAllocator::Simd())
, m_trace(rhs.m_trace)
, m_internal(new Impl(this))
, m_shape(rhs.m_shape)
, m_allocation_info(rhs.m_allocation_info)
{
  m_value = rhs.m_value.constSpan();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ArrayDataT<DataType>::
ArrayDataT(const DataStorageBuildInfo& dsbi)
: m_value(dsbi.memoryAllocator())
, m_trace(dsbi.traceMng())
, m_internal(new Impl(this))
{
  _setShape();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ArrayDataT<DataType>::
~ArrayDataT()
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataStorageTypeInfo ArrayDataT<DataType>::
staticStorageTypeInfo()
{
  typedef DataTypeTraitsT<DataType> TraitsType;
  eBasicDataType bdt = TraitsType::basicDataType();
  Int32 nb_basic_type = TraitsType::nbBasicType();
  Int32 dimension = 1;
  Int32 multi_tag = 0;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataStorageTypeInfo ArrayDataT<DataType>::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Ref<ISerializedData> ArrayDataT<DataType>::
createSerializedDataRef(bool use_basic_type) const
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  Integer nb_count = 1;
  eDataType data_type = dataType();
  Integer type_size = sizeof(DataType);

  if (use_basic_type){
    nb_count = DataTypeTraitsT<DataType>::nbBasicType();
    data_type = DataTypeTraitsT<BasicType>::type();
    type_size = sizeof(BasicType);
  }

  Int64 nb_element = m_value.largeSize();
  Int64 nb_base_element = nb_element * nb_count;
  Int64 full_size = nb_base_element * type_size;
  Span<const Byte> base_values(reinterpret_cast<const Byte*>(m_value.data()),full_size);
  UniqueArray<Int64> extents;
  extents.add(nb_element);
  auto sd = arcaneCreateSerializedDataRef(data_type,base_values.size(),1,nb_element,
                                          nb_base_element,false,extents,shape());
  sd->setConstBytes(base_values);
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  eDataType data_type = sdata->baseDataType();
  eDataType base_data_type = DataTypeTraitsT<BasicType>::type();

  if (data_type!=dataType() && data_type==base_data_type)
    ARCANE_THROW(ArgumentException,"Bad serialized type");

  Int64 nb_element = sdata->nbElement();

  //m_trace->info() << " ASSIGN DATA nb_element=" << nb_element
  //                << " this=" << this;
  m_value.resize(nb_element);
  m_shape = sdata->shape();
  Byte* byte_data = reinterpret_cast<Byte*>(m_value.data());
  Span<Byte> buffer(byte_data,sdata->memorySize());
  sdata->setWritableBytes(buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
assignSerializedData(const ISerializedData* sdata)
{
  ARCANE_UNUSED(sdata);
  // Rien à faire car \a sdata pointe directement vers m_value
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
serialize(ISerializer* sbuf,IDataOperation* operation)
{
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();
  bool is_debug = arcaneIsDebug();

  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    {
      Int64 nb_value = m_value.largeSize();
      Int64 total_size = nb_value*nb_count;
      m_trace->debug(Trace::High) << " ArrayDataT::serialize (full) reserve datatype="
                                  << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      sbuf->reserve(DT_Int64,2); // 1 pour magic number et 1 pour la taille
      sbuf->reserveSpan(data_type,total_size);
    }
    break;
  case ISerializer::ModePut:
    {
      Int64 nb_value = m_value.largeSize();
      Int64 total_size = nb_value*nb_count;
      m_trace->debug(Trace::High) << " ArrayDataT::serialize (full) put datatype="
                                  << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      if (is_debug)
        for( Int64 i=0; i<nb_value; ++i )
          m_trace->debug(Trace::Highest) << "Put i=" << i << " value =" << m_value[i];

      Span<const BasicType> base_value(reinterpret_cast<BasicType*>(m_value.data()),total_size);
      sbuf->putInt64(SERIALIZE_MAGIC_NUMBER);
      sbuf->putInt64(nb_value);
      sbuf->putSpan(base_value);
    }
    break;
  case ISerializer::ModeGet:
    {
      Int64 saved_magic_number = sbuf->getInt64();
      Int64 nb_value = sbuf->getInt64();

      if (saved_magic_number!=SERIALIZE_MAGIC_NUMBER)
        ARCANE_FATAL("Internal errror: bad magic number for serialisation expected={0} current={1}",
                     SERIALIZE_MAGIC_NUMBER,saved_magic_number);
      
      Int64 total_size = nb_value * nb_count;

      m_trace->debug(Trace::High) << " ArrayDataT::serialize (full) get mode=" << sbuf->readMode() 
                                  << " datatype=" << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      switch(sbuf->readMode()){
      case ISerializer::ReadReplace:
        {
          m_value.resize(nb_value); // must resize using resizeFromGroup ?

          if (operation) {
            UniqueArray<BasicType> base_value(total_size);
            sbuf->getSpan(base_value);
            Span<const DataType> data_value(reinterpret_cast<DataType*>(base_value.data()),nb_value);
            operation->applySpan(m_value,data_value);
            if (is_debug)
              for( Int64 i=0; i<nb_value; ++i )
                m_trace->debug(Trace::Highest) << "Get i=" << i << " value="
                                               << data_value[i] << " transformed value=" << m_value[i];
          }
          else{
            Span<BasicType> base_value(reinterpret_cast<BasicType*>(m_value.data()), total_size);
            sbuf->getSpan(base_value);
            if (is_debug)
              for( Int64 i=0; i<nb_value; ++i )
                m_trace->debug(Trace::Highest) << "Get i=" << i << " value=" << m_value[i];
          }
        }
        break;
      case ISerializer::ReadAdd:
        {
          Int64 current_size = m_value.largeSize();
          m_value.resize(current_size+nb_value); // must resize using resizeFromGroup ?
          Span<BasicType> base_value(reinterpret_cast<BasicType*>(m_value.data()+current_size),total_size);
          if (operation)
            ARCANE_THROW(NotImplementedException,"ArrayData::serialize : Cannot deserialize using operation in ReadAdd mode");
          sbuf->getSpan(base_value);
          if (is_debug)
            for( Int64 i=0; i<nb_value; ++i )
              m_trace->debug(Trace::Highest) << "Get i=" << i << " value=" << m_value[i];
        }
      }
      break;
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation)
{
  _serialize(sbuf,ids,operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
_serialize(ISerializer* sbuf,Span<const Int32> ids,IDataOperation* operation)
{
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();
  bool is_debug = arcaneIsDebug();

  Int64 nb_value = ids.size();
  Int64 total_size = nb_value*nb_count;

  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    {
      m_trace->debug(Trace::High) << " ArrayDataT::serialize (partial) reserve datatype="
                                  << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      sbuf->reserve(DT_Int64,2);
      sbuf->reserveSpan(data_type,total_size);
    }
    break;
  case ISerializer::ModePut:
    {
      m_trace->debug(Trace::High) << " ArrayDataT::serialize (partial) put datatype="
                                  << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      if (is_debug){
        // Vérifie les valeurs
        for( Integer i=0, max_value=m_value.size(); i<nb_value; ++i )
          if (ids[i]>max_value)
            throw IndexOutOfRangeException(A_FUNCINFO,
                         String::format(" put,serialize : bad sizes i={0} ids[i]={1} nb_value={2} this={3}",
                                        i,ids[i], max_value, this),
                         i,0,max_value);
      }

      UniqueArray<BasicType> base_value; 
      base_value.reserve(total_size);
      for( Int64 i=0; i<nb_value; ++i ){
#ifdef ARCANE_DEBUG
        m_trace->debug(Trace::Highest) << "Put i=" << i << " index=" << ids[i] << " value=" << m_value[ids[i]];
#endif /* ARCANE_DEBUG */
        ConstArrayView<BasicType> current_value(nb_count, reinterpret_cast<BasicType*>(&m_value[ids[i]]));
        base_value.addRange(current_value);
      }
      sbuf->putInt64(SERIALIZE_MAGIC_NUMBER);
      sbuf->putInt64(nb_value);
      sbuf->putSpan(base_value);
    }
    break;
  case ISerializer::ModeGet:
    {
      m_trace->debug(Trace::High) << " ArrayDataT::serialize (partial) get mode=" << sbuf->readMode() 
                                  << " datatype=" << dataTypeName(data_type)
                                  << " ids=" << nb_value << " totalsize=" << total_size;
      if (is_debug){
        // Vérifie les valeurs
        for( Integer i=0, max_value=m_value.size(); i<nb_value; ++i )
          if (ids[i]>max_value)
            throw IndexOutOfRangeException(A_FUNCINFO,
                         String::format(" put,serialize : bad sizes i={0} ids[i]={1} nb_value={2} this={3}",
                                        i,ids[i], max_value, this),
                         i,0,max_value);
      }

      switch(sbuf->readMode()){
      case ISerializer::ReadReplace:
        {
          Int64 saved_magic_number = sbuf->getInt64();
          Int64 saved_nb_value = sbuf->getInt64();

          if (saved_magic_number!=SERIALIZE_MAGIC_NUMBER)
            ARCANE_FATAL("Internal errror: bad magic number for serialisation expected={0} current={1}",
                         SERIALIZE_MAGIC_NUMBER,saved_magic_number);

          if (saved_nb_value!=nb_value)
            ARCANE_FATAL("Internal errror: bad size for serialisation expected={0} found={1}",
                         nb_value,saved_nb_value);

          UniqueArray<BasicType> base_value(total_size);
          sbuf->getSpan(base_value);

          Span<DataType> data_value(reinterpret_cast<DataType*>(base_value.data()),nb_value);
          UniqueArray<DataType> current_value;
          Span<DataType> transformed_value;

          if (operation && nb_value!=0) {
            current_value.resize(ids.size());
            Arccore::sampleSpan(m_value.constSpan(),ids,current_value.span());
            transformed_value = current_value.view();
            operation->applySpan(transformed_value,data_value);
          }
          else {
            transformed_value = data_value;
          }

          if (is_debug){
            if (operation)
              for( Int64 i=0; i<nb_value; ++i )
                m_trace->debug(Trace::Highest) << "Get i=" << i << " index=" << ids[i]
                                               << " value=" << data_value[i] << " transformed value=" << transformed_value[i];
            else
              for( Int64 i=0; i<nb_value; ++i )
                m_trace->debug(Trace::Highest) << "Get i=" << i << " index=" << ids[i]
                                               << " value=" << data_value[i];
          }

          for( Int64 i=0; i<nb_value; ++i ) {
            m_value[ids[i]] = transformed_value[i];
          }
        }
        break;
      case ISerializer::ReadAdd:
        ARCANE_THROW(NotImplementedException,"ArrayData::serialize : Cannot deserialize with ReadAdd mode");
        break;
      }
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
fillDefault()
{
  m_value.fill(DataType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
setName(const String& name)
{
  m_value.setDebugName(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
computeHash(IHashAlgorithm* algo,ByteArray& output) const
{
  Int64 type_size = sizeof(DataType);
  Int64 nb_element = m_value.largeSize();
  const Byte* ptr = reinterpret_cast<const Byte*>(m_value.data());
  Span<const Byte> input(ptr,type_size*nb_element);
  algo->computeHash64(input,output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
computeHash(DataHashInfo& hash_info) const
{
  hash_info.setVersion(2);
  hash_info.context()->updateHash(asBytes(m_value.span()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
copy(const IData* data)
{
  auto* true_data = dynamic_cast< const DataInterfaceType* >(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException,"Can not cast 'IData' to 'IArrayDataT'");
  m_value.copy(true_data->view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException,"Can not cast 'IData' to 'ArrayDataT'");
  swapValuesDirect(true_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
swapValuesDirect(ThatClass* true_data)
{
  m_value.swap(true_data->m_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
_setShape()
{
  m_shape.setNbDimension(1);
  m_shape.setDimension(0,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void ArrayDataT<DataType>::
setAllocationInfo(const DataAllocationInfo& v)
{
  if (m_allocation_info==v)
    return;
  m_allocation_info = v;
  m_value.setMemoryLocationHint(v.memoryLocationHint());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerArrayDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<ArrayDataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int16>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int32>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int64>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real2>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real3>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real2x2>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real3x3>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ArrayDataT<Byte>;
template class ArrayDataT<Real>;
template class ArrayDataT<Int16>;
template class ArrayDataT<Int32>;
template class ArrayDataT<Int64>;
template class ArrayDataT<Real2>;
template class ArrayDataT<Real2x2>;
template class ArrayDataT<Real3>;
template class ArrayDataT<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namesapce Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
