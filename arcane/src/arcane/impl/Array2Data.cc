// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Data.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array2'.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"
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
  const Int64 SERIALIZE2_MAGIC_NUMBER = 0x12ff7789;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Donnée tableau bi-dimensionnel d'un type \a DataType
 */
template <class DataType>
class Array2DataT
: public ReferenceCounterImpl
, public IArray2DataT<DataType>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
  class Impl;
  friend class Impl;

 public:

  typedef Array2DataT<DataType> ThatClass;
  typedef IArray2DataT<DataType> DataInterfaceType;

 public:

  explicit Array2DataT(ITraceMng* trace);
  explicit Array2DataT(const DataStorageBuildInfo& dsbi);
  Array2DataT(const Array2DataT<DataType>& rhs);
  ~Array2DataT() override;

 public:

  Integer dimension() const override { return 2; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  Array2<DataType>& value() override { return m_value; }
  const Array2<DataType>& value() const override { return m_value; }
  Array2View<DataType> view() override { return m_value; }
  ConstArray2View<DataType> view() const override { return m_value; }
  void resize(Integer new_size) override;
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
  void computeHash(DataHashInfo& hash_algo) const;
  ArrayShape shape() const override { return m_shape; }
  void setShape(const ArrayShape& new_shape) override { m_shape = new_shape; }
  void setAllocationInfo(const DataAllocationInfo& v) override;
  DataAllocationInfo allocationInfo() const override { return m_allocation_info; }

  void visit(IArray2DataVisitor* visitor)
  {
    visitor->applyVisitor(this);
  }
  void visit(IDataVisitor* visitor) override
  {
    visitor->applyDataVisitor(this);
  }
  void visitScalar(IScalarDataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit scalar data with array2 data");
  }
  void visitArray(IArrayDataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit array data with array2 data");
  }
  void visitArray2(IArray2DataVisitor* visitor) override
  {
    visitor->applyVisitor(this);
  }

 public:

  void swapValuesDirect(ThatClass* true_data);
  void changeAllocator(const MemoryAllocationOptions& alloc_info);

 public:

  IArray2DataInternalT<DataType>* _internal() override { return m_internal; }
  IDataInternal* _commonInternal() override { return m_internal; }

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  UniqueArray2<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;
  IArray2DataInternalT<DataType>* m_internal;
  ArrayShape m_shape;
  DataAllocationInfo m_allocation_info;

 private:

  IArray2DataT<DataType>* _cloneTrue() const { return new ThatClass(*this); }
  IArray2DataT<DataType>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class Array2DataT<DataType>::Impl
: public IArray2DataInternalT<DataType>
, public INumericDataInternal
{
 public:

  explicit Impl(Array2DataT<DataType>* p) : m_p(p){}

 public:

  void reserve(Integer new_capacity) override { m_p->m_value.reserve(new_capacity); }
  void resizeOnlyDim1(Int32 new_dim1_size) override
  {
    m_p->m_value.resize(new_dim1_size,m_p->m_value.dim2Size());
  }
  void resize(Int32 new_dim1_size, Int32 new_dim2_size) override
  {
    if (new_dim1_size < 0)
      ARCANE_FATAL("Bad value '{0}' for dim1_size", new_dim1_size);
    if (new_dim2_size < 0)
      ARCANE_FATAL("Bad value '{0}' for dim2_size", new_dim2_size);
    // Cette méthode est appelée si on modifie la deuxième dimension.
    // Dans ce cas cela invalide l'ancienne valeur de shape.
    bool need_reshape = false;
    if (new_dim2_size != m_p->m_value.dim2Size())
      need_reshape = true;
    m_p->m_value.resize(new_dim1_size, new_dim2_size);
    if (need_reshape) {
      m_p->m_shape.setNbDimension(1);
      m_p->m_shape.setDimension(0, new_dim2_size);
    }
  }
  Array2<DataType>& _internalDeprecatedValue() override { return m_p->m_value; }
  void shrink() const override { m_p->m_value.shrink(); }
  bool compressAndClear(DataCompressionBuffer& buf) override
  {
    IDataCompressor* compressor = buf.m_compressor;
    if (!compressor)
      return false;
    Span<const DataType> values = m_p->m_value.to1DSpan();
    Span<const std::byte> bytes = asBytes(values);
    compressor->compress(bytes,buf.m_buffer);
    buf.m_original_dim1_size = m_p->m_value.dim1Size();
    buf.m_original_dim2_size = m_p->m_value.dim2Size();
    m_p->m_value.clear();
    m_p->m_value.shrink();
    return true;
  }
  bool decompressAndFill(DataCompressionBuffer& buf) override
  {
    IDataCompressor* compressor = buf.m_compressor;
    if (!compressor)
      return false;
    m_p->m_value.resize(buf.m_original_dim1_size,buf.m_original_dim2_size);
    Span<DataType> values = m_p->m_value.to1DSpan();
    compressor->decompress(buf.m_buffer,asWritableBytes(values));
    return true;
  }

  MutableMemoryView memoryView() override
  {
    Array2View<DataType> value = m_p->view();
    Int32 dim1_size = value.dim1Size();
    Int32 dim2_size = value.dim2Size();
    DataStorageTypeInfo storage_info = m_p->storageTypeInfo();
    Int32 nb_basic_element = storage_info.nbBasicElement();
    Int32 datatype_size = basicDataTypeSize(storage_info.basicDataType()) * nb_basic_element;
    return makeMutableMemoryView(value.data(), datatype_size * dim2_size, dim1_size);
  }
  Int32 extent0() const override
  {
    return m_p->view().dim1Size();
  }
  INumericDataInternal* numericData() override { return this; }
  void changeAllocator(const MemoryAllocationOptions& v) override { m_p->changeAllocator(v); }
  void computeHash(DataHashInfo& hash_info) override
  {
    m_p->computeHash(hash_info);
  }

 private:

  Array2DataT<DataType>* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Array2DataT<DataType>::
Array2DataT(ITraceMng* trace)
: m_value(AlignedMemoryAllocator::Simd())
, m_trace(trace)
, m_internal(new Impl(this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Array2DataT<DataType>::
Array2DataT(const Array2DataT<DataType>& rhs)
: m_value(AlignedMemoryAllocator::Simd())
, m_trace(rhs.m_trace)
, m_internal(new Impl(this))
, m_allocation_info(rhs.m_allocation_info)
{
  m_value = rhs.m_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Array2DataT<DataType>::
Array2DataT(const DataStorageBuildInfo& dsbi)
: m_value(dsbi.memoryAllocator())
, m_trace(dsbi.traceMng())
, m_internal(new Impl(this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Array2DataT<DataType>::
~Array2DataT()
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataStorageTypeInfo Array2DataT<DataType>::
staticStorageTypeInfo()
{
  typedef DataTypeTraitsT<DataType> TraitsType;
  eBasicDataType bdt = TraitsType::basicDataType();
  Int32 nb_basic_type = TraitsType::nbBasicType();
  Int32 dimension = 2;
  Int32 multi_tag = 0;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataStorageTypeInfo Array2DataT<DataType>::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
resize(Integer new_size)
{
  m_value.resize(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Ref<ISerializedData> Array2DataT<DataType>::
createSerializedDataRef(bool use_basic_type) const
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  Int64 nb_count = 1;
  eDataType data_type = dataType();
  Int64 type_size = sizeof(DataType);

  if (use_basic_type){
    nb_count = DataTypeTraitsT<DataType>::nbBasicType();
    data_type = DataTypeTraitsT<BasicType>::type();
    type_size = sizeof(BasicType);
  }

  Int64 nb_element = m_value.totalNbElement();
  Int64 nb_base_element = nb_element * nb_count;
  Int64 full_size = nb_base_element * type_size;
  const Byte* bt = reinterpret_cast<const Byte*>(m_value.to1DSpan().data());
  Span<const Byte> base_values(bt,full_size);
  UniqueArray<Int64> dimensions;
  dimensions.resize(2);
  dimensions[0] = m_value.dim1Size();
  dimensions[1] = m_value.dim2Size();
    
  auto sd = arcaneCreateSerializedDataRef(data_type,base_values.size(),2,nb_element,
                                          nb_base_element,false,dimensions,shape());
  sd->setConstBytes(base_values);
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;

  eDataType data_type = sdata->baseDataType();
  eDataType base_data_type = DataTypeTraitsT<BasicType>::type();

  if (data_type!=dataType() && data_type==base_data_type)
    ARCANE_FATAL("Bad serialized type");
  bool is_multi_size = sdata->isMultiSize();
  if (is_multi_size)
    ARCANE_FATAL("Can not allocate multi-size array");

  Int64 dim1_size = sdata->extents()[0];
  Int64 dim2_size = sdata->extents()[1];
  //m_trace->info() << " ASSIGN DATA dim1=" << dim1_size
  //                << " dim2=" << dim2_size
  //                << " addr=" << m_value.viewAsArray().unguardedBasePointer();

  m_value.resize(dim1_size,dim2_size);
  m_shape = sdata->shape();

  Byte* byte_data = reinterpret_cast<Byte*>(m_value.to1DSpan().data());
  Span<Byte> bytes_view(byte_data,sdata->memorySize());
  sdata->setWritableBytes(bytes_view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
assignSerializedData(const ISerializedData* sdata)
{
  ARCANE_UNUSED(sdata);
  // Rien à faire car \a sdata pointe directement vers m_value
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
serialize(ISerializer* sbuf,IDataOperation* operation)
{
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eBasicDataType data_type = DataTypeTraitsT<BasicType>::basicDataType();

  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre d'éléments de la première dimension
    // - le nombre d'éléments de la deuxième dimension
    // - le nombre d'éléments de ids.
    // - le nombre magique pour verification
    sbuf->reserveSpan(eBasicDataType::Int64,4);
    // Réserve la mémoire pour les valeurs
    Int64 total_nb_element = m_value.totalNbElement();
    sbuf->reserveSpan(data_type,total_nb_element*nb_count);
  }
  else if (mode==ISerializer::ModePut){
    Int64 count = m_value.dim1Size();
    Int64 total = m_value.totalNbElement();
    Int64 n[4];
    n[0] = count;
    n[1] = m_value.dim2Size();
    n[2] = total;
    n[3] = SERIALIZE2_MAGIC_NUMBER;
    sbuf->putSpan(Span<const Int64>(n,4));
    BasicType* bt = reinterpret_cast<BasicType*>(m_value.to1DSpan().data());
    Span<const BasicType> v(bt,total*nb_count);
    //m_trace->info() << "PUT array nb_elem=" << (total*nb_count) << " sizeof=" << sizeof(BasicType);
    sbuf->putSpan(v);
  }
  else if (mode==ISerializer::ModeGet){
    Int64 n[4] = { 0, 0, 0, 0 };
    sbuf->getSpan(Span<Int64>(n,4));
    Int64 count = n[0];
    Int64 dim2_size = n[1];
    Int64 total = n[2];
    if (n[3]!=SERIALIZE2_MAGIC_NUMBER)
      ARCANE_FATAL("Bad magic number");
    switch(sbuf->readMode()){
    case ISerializer::ReadReplace:
      {
        //m_trace->info() << "READ REPLACE count=" << count << " dim2_size=" << dim2_size;
        m_value.resize(count,dim2_size);
        if (operation)
          throw NotImplementedException(A_FUNCINFO,"serialize(ReadReplace) with IDataOperation");
        BasicType* bt = reinterpret_cast<BasicType*>(m_value.to1DSpan().data());
        Span<BasicType> v(bt,total*nb_count);
        sbuf->getSpan(v);
      }
      break;
    case ISerializer::ReadAdd:
      {
        Int64 current_size = m_value.dim1Size();
        Int64 current_total = m_value.totalNbElement();
        //m_trace->info() << "READ ADD NEW_SIZE=" << current_size << " COUNT=" << count
        //                << " dim2_size=" << dim2_size << " current_dim2_size=" << m_value.dim2Size()
        //                << " current_total=" << current_total << " read_elem=" << (total*nb_count);
        m_value.resize(current_size + count,dim2_size);
        if (operation)
          throw NotImplementedException(A_FUNCINFO,"serialize(ReadAdd) with IDataOperation");
        BasicType* bt = reinterpret_cast<BasicType*>(m_value.to1DSpan().data()+current_total);
        //m_trace->info() << "GET array nb_elem=" << (total*nb_count) << " sizeof=" << sizeof(BasicType);
        Span<BasicType> v(bt,total*nb_count);
        sbuf->getSpan(v);
      }
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation)
{
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eBasicDataType data_type = DataTypeTraitsT<BasicType>::basicDataType();

  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre d'éléments de la première dimension
    // - le nombre d'éléments de la deuxième dimension
    // - le nombre d'éléments de ids.
    // - le nombre magique pour verification
    sbuf->reserveSpan(eBasicDataType::Int64,4);
    // Réserve la mémoire pour les valeurs
    Int64 total_nb_value = ((Int64)ids.size()) * ((Int64)m_value.dim2Size());
    sbuf->reserveSpan(data_type,total_nb_value*nb_count);
    
  }
  else if (mode==ISerializer::ModePut){
    Int32 count = ids.size();
    Int64 dim2_size =  m_value.dim2Size();
    Int64 total_nb_value = count * dim2_size;
    Int64 total = total_nb_value;
    Int64 n[4];
    n[0] = m_value.dim1Size();
    n[1] = m_value.dim2Size();
    n[2] = count;
    n[3] = SERIALIZE2_MAGIC_NUMBER;
    /*m_trace->info() << "PUT COUNT = " << count << " total=" << total
                    << " dim1 (n[0])=" << n[0]
                    << " dim2 (n[1])=" << n[1]
                    << " count (n[2])=" << n[2]
                    << " magic=" << n[3]
                    << " this=" << this;*/
    sbuf->putSpan(Span<const Int64>(n,4));
    UniqueArray<BasicType> v(total*nb_count);
    {
      Integer index = 0;
      for( Int32 i=0, is=count; i<is; ++i ){
        const BasicType* sub_a = reinterpret_cast<const BasicType*>(m_value[ids[i]].data());
        for( Int64 z=0, iz=dim2_size*nb_count; z<iz; ++z ){
          v[index] = sub_a[z];
          ++index;
        }
      }
    }
    sbuf->putSpan(v);
  }
  else if (mode==ISerializer::ModeGet){
    switch(sbuf->readMode()){
    case ISerializer::ReadReplace:
      {
        Int64 n[4] = { 0, 0, 0, 0 };
        sbuf->getSpan(Span<Int64>(n,4));
        //Integer dim1_size = n[0];
        Int64 dim2_size = n[1];
        Int32 count = CheckedConvert::toInt32(n[2]);
        Int64 total = count * dim2_size;
        // One dim
        /*m_trace->info() << "COUNT = " << count << " total=" << total
                        << " dim1 (n[0])=" << n[0]
                        << " dim1 current=" << m_value.dim1Size()
                        << " dim2 (n[1])=" << n[1]
                        << " dim2 current=" << m_value.dim2Size()
                        << " count (n[2])=" << n[2]
                        << " magic=" << n[3]
                        << " this=" << this;*/
        if (n[3]!=SERIALIZE2_MAGIC_NUMBER)
          ARCANE_FATAL("Bad magic number");
        Int64 current_dim2_size = m_value.dim2Size();
        if (dim2_size!=current_dim2_size){
          if (current_dim2_size!=0 && dim2_size!=0)
            ARCANE_FATAL("serialized data should have the same dim2Size current={0} found={1}",
                         current_dim2_size,dim2_size);
          else
            m_value.resize(m_value.dim1Size(),dim2_size);
        }
        Int64 nb_value = count;
        //Array<BasicType> v(total*nb_count);
        UniqueArray<BasicType> base_value(total*nb_count);

        sbuf->getSpan(base_value);

        Span<DataType> data_value(reinterpret_cast<DataType*>(base_value.data()),nb_value*dim2_size);
        UniqueArray<DataType> current_value;
        Span<DataType> transformed_value;

        // Si on applique une transformantion, effectue la transformation dans un
        // tableau temporaire 'current_value'.
        if (operation && nb_value!=0) {
          current_value.resize(data_value.size());

          Int64 index = 0;
          for( Int32 i=0, n=count; i<n; ++i ){
            Span<const DataType> a(m_value[ids[i]]);
            for( Int64 z=0, iz=dim2_size; z<iz; ++z ){
              current_value[index] = a[z];
              ++index;
            }
          }

          transformed_value = current_value.view();
          operation->applySpan(transformed_value,data_value);
        }
        else {
          transformed_value = data_value;
        }

        {
          Int64 index = 0;
          for( Int32 i=0, n=count; i<n; ++i ){
            Span<DataType> a(m_value[ids[i]]);
            for( Int64 z=0, iz=dim2_size; z<iz; ++z ){
              a[z] = transformed_value[index];
              ++index;
            }
          }
        }
      }
      break;
    case ISerializer::ReadAdd:
      throw NotImplementedException(A_FUNCINFO,"option 'ReadAdd'");
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
fillDefault()
{
  m_value.fill(DataType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
setName(const String& name)
{
  m_value.setDebugName(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
computeHash(IHashAlgorithm* algo,ByteArray& output) const
{
  //TODO: passer en 64 bits
  Span<const DataType> values = m_value.to1DSpan();

  // Calcule la fonction de hashage pour les valeurs
  Int64 type_size = sizeof(DataType);
  Int64 nb_element = values.size();
  const Byte* ptr = reinterpret_cast<const Byte*>(values.data());
  Span<const Byte> input(ptr,type_size*nb_element);
  algo->computeHash64(input,output);

  // Calcule la fonction de hashage pour les tailles
  UniqueArray<Int64> dimensions(2);
  dimensions[0] = m_value.dim1Size();
  dimensions[1] = m_value.dim2Size();
  ptr = reinterpret_cast<const Byte*>(dimensions.data());
  Int64 array_len = dimensions.size() * sizeof(Int64);
  input = Span<const Byte>(ptr,array_len);
  algo->computeHash64(input,output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
computeHash(DataHashInfo& hash_info) const
{
  hash_info.setVersion(2);
  IHashAlgorithmContext* context = hash_info.context();

  // Calcule la fonction de hashage pour les tailles
  Int64 dimensions[2];
  dimensions[0] = m_value.dim1Size();
  dimensions[1] = m_value.dim2Size();
  Span<const Int64> dimension_span(dimensions,2);
  context->updateHash(asBytes(dimension_span));

  // Calcule la fonction de hashage pour les valeurs
  context->updateHash(asBytes(m_value.to1DSpan()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
copy(const IData* data)
{
  auto* true_data = dynamic_cast< const DataInterfaceType* >(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'IArray2DataT'");
  m_value.copy(true_data->view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'Array2DataT'");
  swapValuesDirect(true_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
swapValuesDirect(ThatClass* true_data)
{
  m_value.swap(true_data->m_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
setAllocationInfo(const DataAllocationInfo& v)
{
  if (m_allocation_info==v)
    return;
  m_allocation_info = v;
  m_value.setMemoryLocationHint(v.memoryLocationHint());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2DataT<DataType>::
changeAllocator(const MemoryAllocationOptions& alloc_info)
{
  ARCANE_UNUSED(alloc_info);
  ARCANE_THROW(NotImplementedException,"changeAllocator for 2D Array");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerArray2DataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<Array2DataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<BFloat16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Float16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Float32>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int8>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int32>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int64>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real2>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real3>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real2x2>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real3x3>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class Array2DataT<Byte>;
template class Array2DataT<Real>;
template class Array2DataT<Float16>;
template class Array2DataT<BFloat16>;
template class Array2DataT<Float32>;
template class Array2DataT<Int8>;
template class Array2DataT<Int16>;
template class Array2DataT<Int32>;
template class Array2DataT<Int64>;
template class Array2DataT<Real2>;
template class Array2DataT<Real2x2>;
template class Array2DataT<Real3>;
template class Array2DataT<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
