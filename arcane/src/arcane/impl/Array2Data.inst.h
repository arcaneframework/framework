// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Data.inst.h                                           (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array2'.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/Array2Data.h"

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
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/datatype/DataStorageBuildInfo.h"
#include "arcane/core/datatype/IDataOperation.h"

#include "arcane/core/ISerializer.h"

#include "arcane/impl/SerializedData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace 
{
inline constexpr Int64 SERIALIZE2_MAGIC_NUMBER = 0x12ff7789;
}

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
  // Il faut utiliser par resizeNoInit() car si la mémoire demandée
  // est le device on ne peut pas utiliser les constructeurs si le type
  // n'est pas un type basique car l'opération est faite côté CPU.
  UniqueArray2<DataType> new_value(alloc_info.allocator());
  new_value.resizeNoInit(m_value.dim1Size(), m_value.dim2Size());

  // Copie \a m_value dans \a new_value
  // Tant qu'il n'y a pas l'API dans Arccore, il faut faire la copie à la
  // main pour ne pas avoir de plantage si l'allocateur est uniquement sur
  // un accélérateur
  MemoryUtils::copy(new_value.to1DSpan(), Span<const DataType>(m_value.to1DSpan()));

  std::swap(m_value, new_value);
  m_allocation_info.setMemoryLocationHint(alloc_info.memoryLocationHint());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
