﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
template <class DataType,int RankValue>
class INumArrayDataT
: public IData
{
 public:

  typedef INumArrayDataT<DataType,RankValue> ThatClass;

 public:

  //! Vue constante sur la donnée
  virtual MDSpan<const DataType,RankValue> view() const = 0;

  //! Vue sur la donnée
  virtual MDSpan<DataType,RankValue> view() = 0;

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
template <class DataType,int RankValue>
class NumArrayDataT
: public ReferenceCounterImpl
, public INumArrayDataT<DataType,RankValue>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
  class Impl;
  friend Impl;

 public:

  typedef NumArrayDataT<DataType,RankValue> ThatClass;
  typedef INumArrayDataT<DataType,RankValue> DataInterfaceType;

 public:

  explicit NumArrayDataT(ITraceMng* trace);
  explicit NumArrayDataT(const DataStorageBuildInfo& dsbi);
  NumArrayDataT(const NumArrayDataT<DataType,RankValue>& rhs);
  ~NumArrayDataT() override;

 public:

  Integer dimension() const override { return 2; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  //NumArray<DataType,RankValue>& value() override { return m_value; }
  //const NumArray<DataType,RankValue>& value() const override { return m_value; }
  MDSpan<DataType,RankValue> view() override { return m_value.span(); }
  MDSpan<const DataType,RankValue> view() const override { return m_value.span(); }
  void resize(Integer new_size) override;
  IData* clone() override { return _cloneTrue(); }
  IData* cloneEmpty() override { return _cloneTrueEmpty(); };
  Ref<IData> cloneRef() override { return makeRef(_cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(_cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
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

  NumArray<DataType,RankValue> m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  INumArrayDataT<DataType,RankValue>* _cloneTrue() const { return new ThatClass(*this); }
  INumArrayDataT<DataType,RankValue>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
  void _resizeDim1(Int32 dim1_size);
  Int64 _getDim2Size() const;
  Span2<DataType> _valueAsSpan2();
  Span2<const DataType> _valueAsConstSpan2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> NumArrayDataT<DataType,RankValue>::
NumArrayDataT(ITraceMng* trace)
: m_trace(trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> NumArrayDataT<DataType,RankValue>::
NumArrayDataT(const NumArrayDataT<DataType,RankValue>& rhs)
: m_value(rhs.m_value)
, m_trace(rhs.m_trace)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> NumArrayDataT<DataType,RankValue>::
NumArrayDataT(const DataStorageBuildInfo& dsbi)
: m_trace(dsbi.traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> NumArrayDataT<DataType,RankValue>::
~NumArrayDataT()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> DataStorageTypeInfo NumArrayDataT<DataType,RankValue>::
staticStorageTypeInfo()
{
  typedef DataTypeTraitsT<DataType> TraitsType;
  eBasicDataType bdt = TraitsType::basicDataType();
  Int32 nb_basic_type = TraitsType::nbBasicType();
  Int32 dimension = RankValue;
  Int32 multi_tag = 0;
  String impl_name = "NumArray";
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag,impl_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> DataStorageTypeInfo NumArrayDataT<DataType,RankValue>::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
_resizeDim1(Int32 dim1_size)
{
  // Récupère les dimensions du 'NumArray' et ne modifie que la première
  auto extents = m_value.extents();
  extents.setExtent(0,dim1_size);
  m_value.resize(extents);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> Int64
NumArrayDataT<DataType,RankValue>::
_getDim2Size() const
{
  // Récupère les dimensions du 'NumArray' et considère que 'dim2_size' est
  // le produits du nombre d'éléments des dimensions après la première.
  auto extents = m_value.extents();
  Int64 dim2_size = 1;
  for( Integer i=0; i<RankValue; ++i )
    dim2_size *= extents(i);
  return dim2_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ATTENTION: Ne fonctionnera pas s'il y a des 'strides'
template<typename DataType,int RankValue> Span2<DataType>
NumArrayDataT<DataType,RankValue>::
_valueAsSpan2()
{
  Int64 dim1_size = m_value.extent(0);
  Int64 dim2_size = _getDim2Size();
  Span2<DataType> value_as_span2(m_value.to1DSpan().data(),dim1_size,dim2_size);
  return value_as_span2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ATTENTION: Ne fonctionnera pas s'il y a des 'strides'
// TODO: a supprimer mais pour cela il faut pouvoir convertir un Span<T> en
// un Span<const T> et cela n'est pas encore possible avec arccore.
template<typename DataType,int RankValue> Span2<const DataType>
NumArrayDataT<DataType,RankValue>::
_valueAsConstSpan2()
{
  Int64 dim1_size = m_value.extent(0);
  Int64 dim2_size = _getDim2Size();
  Span2<const DataType> value_as_span2(m_value.to1DSpan().data(),dim1_size,dim2_size);
  return value_as_span2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
resize(Integer new_size)
{
  _resizeDim1(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> Ref<ISerializedData>
NumArrayDataT<DataType,RankValue>::
createSerializedDataRef(bool use_basic_type) const
{
  using BasicType = typename DataTypeTraitsT<DataType>::BasicType;

  Int64 nb_count = 1;
  eDataType data_type = dataType();
  Int64 type_size = sizeof(DataType);

  if (use_basic_type){
    nb_count = 1; //DataTypeTraitsT<DataType>::nbBasicType();
    data_type = DataTypeTraitsT<BasicType>::type();
    type_size = sizeof(BasicType);
  }

  Int64 nb_element = m_value.totalNbElement();
  Int64 nb_base_element = nb_element * nb_count;
  Int64 full_size = nb_base_element * type_size;
  const Byte* bt = reinterpret_cast<const Byte*>(m_value.to1DSpan().data());
  Span<const Byte> base_values(bt,full_size);
  auto extents = m_value.extents().asSpan();
  UniqueArray<Int64> dimensions(extents.size());
  for ( Int32 i=0; i<extents.size(); ++i )
    dimensions[i] = extents[i];
  auto sd = arcaneCreateSerializedDataRef(data_type,base_values.size(),RankValue,nb_element,
                                          nb_base_element,false,dimensions);
  sd->setConstBytes(base_values);
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  using BasicType = typename DataTypeTraitsT<DataType>::BasicType;

  eDataType data_type = sdata->baseDataType();
  eDataType base_data_type = DataTypeTraitsT<BasicType>::type();

  if (data_type!=dataType() && data_type==base_data_type)
    ARCANE_FATAL("Bad serialized type");

  bool is_multi_size = sdata->isMultiSize();
  if (is_multi_size)
    ARCANE_FATAL("Can not allocate multi-size array");

  // Converti en Int32
  Int64ConstArrayView sdata_extents = sdata->extents();
  std::array<Int32,RankValue> numarray_extents;
  for( Int32 i=0; i<RankValue; ++i )
    numarray_extents[i] = CheckedConvert::toInt32(sdata_extents[i]);
  ArrayExtents<RankValue> extents = ArrayExtents<RankValue>::fromSpan(numarray_extents);
  m_value.resize(extents);

  Byte* byte_data = reinterpret_cast<Byte*>(m_value.to1DSpan().data());
  Span<Byte> bytes_view(byte_data,sdata->memorySize());
  sdata->setWritableBytes(bytes_view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
assignSerializedData(const ISerializedData* sdata)
{
  ARCANE_UNUSED(sdata);
  // Rien à faire car \a sdata pointe directement vers m_value
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
serialize(ISerializer* sbuf,IDataOperation* operation)
{
  // NOTE: Cette méthode n'est pas encore opérationnelle

  Integer nb_count = 1; //DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  //eDataType data_type = DataTypeTraitsT<BasicType>::type();

  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre magique pour verification
    // - le nombre d'éléments de ids.
    sbuf->reserveSpan(DT_Int64,2);
    // Réserve la mémoire pour le nombre d'éléments de chaque dimension (soit RankValue)
    sbuf->reserveSpan(DT_Int32,RankValue);
    // Réserve la mémoire pour les valeurs
    sbuf->reserveSpan(m_value.to1DSpan());
  }
  else if (mode==ISerializer::ModePut){
    Int64 total = m_value.totalNbElement();
    Int64 n[2];
    n[0] = SERIALIZE2_MAGIC_NUMBER;
    n[1] = total;
    sbuf->putSpan(Span<const Int64>(n,2));
    sbuf->putSpan(m_value.extents().asSpan());
    sbuf->putSpan(m_value.to1DSpan());
  }
  else if (mode==ISerializer::ModeGet){
    Int64 n[2] = { 0, 0 };
    sbuf->getSpan(Span<Int64>(n,2));
    Int64 total = n[1];
    if (n[0]!=SERIALIZE2_MAGIC_NUMBER)
      ARCANE_FATAL("Bad magic number");
    Int32 extents_buf[RankValue];
    SmallSpan<Int32> extents_span(extents_buf,RankValue);
    sbuf->getSpan(extents_span);
    Int32 count = extents_span[0];
    switch(sbuf->readMode()){
    case ISerializer::ReadReplace:
      {
        //m_trace->info() << "READ REPLACE count=" << count << " dim2_size=" << dim2_size;
        m_value.resize(ArrayExtents<RankValue>::fromSpan(extents_span));
        if (operation)
          ARCANE_THROW(NotImplementedException,"serialize(ReadReplace) with IDataOperation");
        BasicType* bt = reinterpret_cast<BasicType*>(m_value.to1DSpan().data());
        Span<BasicType> v(bt,total*nb_count);
        sbuf->getSpan(v);
      }
      break;
    case ISerializer::ReadAdd:
      {
        Int32 current_size = m_value.extent(0);
        // TODO: vérifier que dim2_size a la même valeur qu'en entrée.
        // Int64 dim2_size = _getDim2Size();
        Int64 current_total = m_value.totalNbElement();
        //m_trace->info() << "READ ADD NEW_SIZE=" << current_size << " COUNT=" << count
        //                << " dim2_size=" << dim2_size << " current_dim2_size=" << m_value.dim2Size()
        //                << " current_total=" << current_total << " read_elem=" << (total*nb_count);
        _resizeDim1(current_size + count);
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

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation)
{
  ARCANE_UNUSED(operation);
  // TODO: mutualiser avec serialize(sbuf,ids);

  [[maybe_unused]] Integer nb_count = 1;
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  eDataType data_type = DataTypeTraitsT<BasicType>::type();
  ISerializer::eMode mode = sbuf->mode();
  if (mode==ISerializer::ModeReserve){
    // Réserve la mémoire pour
    // - le nombre magique pour verification
    // - le nombre d'éléments de ids.
    // - 
    sbuf->reserveSpan(DT_Int64,3);
    // Réserve la mémoire pour le nombre d'éléments de chaque dimension (soit RankValue)
    sbuf->reserveSpan(DT_Int32,RankValue);
    // Réserve la mémoire pour les valeurs
    auto sub_extent = m_value.extents().removeFirstExtent();
    sbuf->reserveSpan(data_type,sub_extent.totalNbElement() * ids.size());
  }
  else if (mode==ISerializer::ModePut){
    Int32 count = ids.size();
    Int64 dim2_size = _getDim2Size();
    Int64 total_nb_value = count * dim2_size;
    Int64 total = total_nb_value;

    Int64 n[3];
    n[0] = SERIALIZE2_MAGIC_NUMBER;
    n[1] = count;
    n[2] = dim2_size;
    /*m_trace->info() << "PUT COUNT = " << count << " total=" << total
                    << " dim1 (n[0])=" << n[0]
                    << " dim2 (n[1])=" << n[1]
                    << " count (n[2])=" << n[2]
                    << " magic=" << n[3]
                    << " this=" << this;*/
    sbuf->putSpan(Span<const Int64>(n,3));

    sbuf->putSpan(m_value.extents().asSpan());

    UniqueArray<BasicType> v(total*nb_count);
    Span2<const DataType> value_as_span2(_valueAsConstSpan2());
    {
      Integer index = 0;
      for( Int32 i=0, n=count; i<n; ++i ){
        const BasicType* sub_a = reinterpret_cast<const BasicType*>(value_as_span2[ids[i]].data());
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
        Int64 n[3] = { 0, 0, 0 };
        sbuf->getSpan(Span<Int64>(n,3));
        Int32 count = CheckedConvert::toInt32(n[1]);
        Int64 dim2_size = n[2];
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
        if (n[1]!=SERIALIZE2_MAGIC_NUMBER)
          ARCANE_FATAL("Bad magic number");

        Int32 extents_buf[RankValue];
        Span<Int32> extents_span(extents_buf,RankValue);
        sbuf->getSpan(extents_span);

        // TODO: utiliser extent pour vérifier que le tableau recu à le
        // même nombre d'éléments dans les dimensions 2+.
        Int64 current_dim2_size = _getDim2Size();

        if (dim2_size!=current_dim2_size){
          if (current_dim2_size!=0 && dim2_size!=0)
            ARCANE_FATAL("serialized data should have the same dim2Size current={0} found={1}",
                         current_dim2_size,dim2_size);
          else
            _resizeDim1(m_value.dim1Size());
        }
        Int64 nb_value = count;
        //Array<BasicType> v(total*nb_count);
        UniqueArray<BasicType> base_value(total*nb_count);

        sbuf->getSpan(base_value);

        Span<DataType> data_value(reinterpret_cast<DataType*>(base_value.data()),nb_value*dim2_size);
        UniqueArray<DataType> current_value;
        Span<DataType> transformed_value;

        Span2<DataType> value_as_span2(_valueAsSpan2());
        // Si on applique une transformantion, effectue la transformation dans un
        // tableau temporaire 'current_value'.
        if (operation && nb_value!=0) {
          current_value.resize(data_value.size());

          Int64 index = 0;
          for( Int32 i=0, n=count; i<n; ++i ){
            Span<const DataType> a(value_as_span2[ids[i]]);
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
            Span<DataType> a(value_as_span2[ids[i]]);
            for( Int64 z=0, iz=dim2_size; z<iz; ++z ){
              a[z] = transformed_value[index];
              ++index;
            }
          }
        }
      }
      break;
    case ISerializer::ReadAdd:
      ARCANE_THROW(NotImplementedException,"option 'ReadAdd'");
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
fillDefault()
{
  m_value.fill(DataType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
setName(const String& name)
{
  ARCANE_UNUSED(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
computeHash(IHashAlgorithm* algo,ByteArray& output) const
{
  // Calcule la fonction de hashage pour les valeurs
  {
    Span<const DataType> values = m_value.to1DSpan();
    Int64 type_size = sizeof(DataType);
    Int64 nb_element = values.size();
    const Byte* ptr = reinterpret_cast<const Byte*>(values.data());
    Span<const Byte> input(ptr,type_size*nb_element);
    algo->computeHash64(input,output);
  }

  {
    // Calcule la fonction de hashage pour les nombres d'éléments
    auto input = asBytes(m_value.extents().asSpan());
    algo->computeHash64(input,output);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
copy(const IData* data)
{
  auto* true_data = dynamic_cast< const DataInterfaceType* >(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'INumArrayDataT'");
  m_value.copy(true_data->view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    throw ArgumentException(A_FUNCINFO,"Can not cast 'IData' to 'NumArrayDataT'");
  swapValuesDirect(true_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int RankValue> void NumArrayDataT<DataType,RankValue>::
swapValuesDirect(ThatClass* true_data)
{
  m_value.swap(true_data->m_value);
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

  DataStorageFactory<NumArrayDataT<Real,3>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int16,2>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int32,2>>::registerDataFactory(dfm);
  DataStorageFactory<NumArrayDataT<Int64,3>>::registerDataFactory(dfm);

  DataStorageFactory<NumArrayDataT<Real,4>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int16,2>>::registerDataFactory(dfm);
  //DataStorageFactory<NumArrayDataT<Int32,2>>::registerDataFactory(dfm);
  DataStorageFactory<NumArrayDataT<Int64,4>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
