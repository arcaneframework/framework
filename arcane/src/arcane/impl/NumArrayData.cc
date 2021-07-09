// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayData.cc                                             (C) 2000-2021 */
/*                                                                           */
/* Donnée de type 'NumArray'.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

#include "arcane/utils/NumArray.h"
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

#include "arcane/datatype/IDataOperation.h"
#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataStorageBuildInfo.h"
#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/ISerializer.h"
#include "arcane/IData.h"
#include "arcane/IDataVisitor.h"

#include "arcane/core/internal/IDataInternal.h"

#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataStorageFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace 
{
  const Int64 SERIALIZE2_MAGIC_NUMBER = 0x923abd20;
}
/*!
 * \brief Interface d'un 'IData' dont le conteneur repose sur un 'NumArray'.
 */
template <class DataType,int Rank>
class INumArrayDataT
: public IData
{
 public:

  typedef INumArrayDataT<DataType,Rank> ThatClass;

 public:

  //! Vue constante sur la donnée
  virtual MDSpan<const DataType,Rank> view() const = 0;

  //! Vue sur la donnée
  virtual MDSpan<DataType,Rank> view() = 0;

  //! Clone la donnée
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone la donnée mais sans éléments.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'un 'IData' dont le conteneur repose sur un 'NumArray'.
 */
template <class DataType,int Rank>
class NumArrayDataT
: public ReferenceCounterImpl
, public INumArrayDataT<DataType,Rank>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
  class Impl;
  friend Impl;

 public:

  typedef NumArrayDataT<DataType,Rank> ThatClass;
  typedef INumArrayDataT<DataType,Rank> DataInterfaceType;

 public:

  explicit NumArrayDataT(ITraceMng* trace);
  explicit NumArrayDataT(const DataStorageBuildInfo& dsbi);
  NumArrayDataT(const NumArrayDataT<DataType,Rank>& rhs);
  ~NumArrayDataT() override;

 public:

  Integer dimension() const override { return 2; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  //NumArray<DataType,Rank>& value() override { return m_value; }
  //const NumArray<DataType,Rank>& value() const override { return m_value; }
  MDSpan<DataType,Rank> view() override { return m_value.span(); }
  MDSpan<const DataType,Rank> view() const override { return m_value.span(); }
  void resize(Integer new_size) override;
  IData* clone() override { return _cloneTrue(); }
  IData* cloneEmpty() override { return _cloneTrueEmpty(); }
  Ref<IData> cloneRef() override { return makeRef(_cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(_cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
  Ref<DataInterfaceType> cloneTrueRef() override { auto* d = _cloneTrue(); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { auto* d = _cloneTrueEmpty(); return makeRef(d); }
  void fillDefault() override;
  void setName(const String& name) override;
  const ISerializedData* createSerializedData(bool use_basic_type) const override;
  Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const override;
  void allocateBufferForSerializedData(ISerializedData* sdata) override;
  void assignSerializedData(const ISerializedData* sdata) override;
  void copy(const IData* data) override;
  void swapValues(IData* data) override;
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
  void visit(IArray2DataVisitor*)
  {
    ARCANE_THROW(NotSupportedException, "Can not visit array2 data with NumArray data");
  }
  void visit(IDataVisitor* visitor) override
  {
    ARCANE_UNUSED(visitor);
    //visitor->applyDataVisitor(this);
    ARCANE_THROW(NotImplementedException,"visit(IDataVisitor*)");
  }
  void visitScalar(IScalarDataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit scalar data with NumArray data");
  }
  void visitArray(IArrayDataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit array data with NumArray data");
  }
  void visitArray2(IArray2DataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit array2 data with NumArray data");
  }
  void visitMultiArray2(IMultiArray2DataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit multiarray2 data with NumArray data");
  }

 public:

  void swapValuesDirect(ThatClass* true_data);

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  NumArray<DataType,Rank> m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  INumArrayDataT<DataType,Rank>* _cloneTrue() const { return new ThatClass(*this); }
  INumArrayDataT<DataType,Rank>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> NumArrayDataT<DataType,Rank>::
NumArrayDataT(ITraceMng* trace)
: m_trace(trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> NumArrayDataT<DataType,Rank>::
NumArrayDataT(const NumArrayDataT<DataType,Rank>& rhs)
: m_value(rhs.m_value)
, m_trace(rhs.m_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> NumArrayDataT<DataType,Rank>::
NumArrayDataT(const DataStorageBuildInfo& dsbi)
: m_trace(dsbi.traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> NumArrayDataT<DataType,Rank>::
~NumArrayDataT()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> DataStorageTypeInfo NumArrayDataT<DataType,Rank>::
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

template<typename DataType,int Rank> DataStorageTypeInfo NumArrayDataT<DataType,Rank>::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
resize(Integer new_size)
{
  ARCANE_UNUSED(new_size);
  ARCANE_THROW(NotImplementedException,"resize()");
#if 0
  m_value.resize(new_size);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> Ref<ISerializedData> NumArrayDataT<DataType,Rank>::
createSerializedDataRef(bool use_basic_type) const
{
  return makeRef(const_cast<ISerializedData*>(createSerializedData(use_basic_type)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> const ISerializedData* NumArrayDataT<DataType,Rank>::
createSerializedData(bool use_basic_type) const
{
  ARCANE_UNUSED(use_basic_type);
  ARCANE_THROW(NotImplementedException,"createSerializedData()");
#if 0
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
    
  ISerializedData* sd = new SerializedData(data_type,base_values.size(),2,nb_element,
                                           nb_base_element,false,dimensions);
  sd->setBytes(base_values);
  return sd;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  ARCANE_UNUSED(sdata);
  ARCANE_THROW(NotImplementedException,"allocateBufferForSerializedData()");
#if 0
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

  Byte* byte_data = reinterpret_cast<Byte*>(m_value.to1DSpan().data());
  Span<Byte> bytes_view(byte_data,sdata->memorySize());
  sdata->setBytes(bytes_view);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
assignSerializedData(const ISerializedData* sdata)
{
  ARCANE_UNUSED(sdata);
  // Rien à faire car \a sdata pointe directement vers m_value
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
serialize(ISerializer* sbuf,IDataOperation* operation)
{
  ARCANE_UNUSED(sbuf);
  ARCANE_UNUSED(operation);
  ARCANE_THROW(NotImplementedException,"serialize()");
#if 0
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();

  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre d'éléments de la première dimension
    // - le nombre d'éléments de la deuxième dimension
    // - le nombre d'éléments de ids.
    // - le nombre magique pour verification
    sbuf->reserveSpan(DT_Int64,4);
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
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation)
{
  ARCANE_UNUSED(sbuf);
  ARCANE_UNUSED(ids);
  ARCANE_UNUSED(operation);
  ARCANE_THROW(NotImplementedException,"serialize()");
#if 0
  Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();

  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre d'éléments de la première dimension
    // - le nombre d'éléments de la deuxième dimension
    // - le nombre d'éléments de ids.
    // - le nombre magique pour verification
    sbuf->reserveSpan(DT_Int64,4);
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

        MDSpan<DataType,Rank> data_value(reinterpret_cast<DataType*>(base_value.data()),nb_value*dim2_size);
        UniqueArray<DataType> current_value;
        MDSpan<DataType,Rank> transformed_value;

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
            MDSpan<DataType,Rank> a(m_value[ids[i]]);
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
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
fillDefault()
{
  m_value.fill(DataType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
setName(const String& name)
{
  ARCANE_UNUSED(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
computeHash(IHashAlgorithm* algo,ByteArray& output) const
{
  ARCANE_UNUSED(algo);
  ARCANE_UNUSED(output);
  ARCANE_THROW(NotImplementedException,"computeHash()");
#if 0
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
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
copy(const IData* data)
{
  ARCANE_UNUSED(data);
  ARCANE_THROW(NotImplementedException,"copy()");
#if 0
  auto* true_data = dynamic_cast< const DataInterfaceType* >(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'INumArrayDataT'");
  m_value.copy(true_data->view());
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'NumArrayDataT'");
  swapValuesDirect(true_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int Rank> void NumArrayDataT<DataType,Rank>::
swapValuesDirect(ThatClass* true_data)
{
  ARCANE_UNUSED(true_data);
  ARCANE_THROW(NotImplementedException,"swapValuesDirect()");
#if 0
  m_value.swap(true_data->m_value);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerNumArrayDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<NumArrayDataT<Real,1>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int16,1>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int32,1>>::registerDataFactory(dfm);
  DataStorageFactory<NumArrayDataT<Int64,1>>::registerDataFactory(dfm);

  DataStorageFactory<NumArrayDataT<Real,2>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int16,2>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int32,2>>::registerDataFactory(dfm);
  DataStorageFactory<NumArrayDataT<Int64,2>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
