// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayData.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array'.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_ARRAYDATA_H
#define ARCANE_IMPL_INTERNAL_ARRAYDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/core/IData.h"
#include "arcane/core/IDataVisitor.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/datatype/DataTypeTraits.h"
#include "arcane/core/datatype/DataStorageTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
    ARCANE_THROW(NotSupportedException, "Can not visit scalar data with array data");
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
  void changeAllocator(const MemoryAllocationOptions& alloc_info);

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
  Int32 extent0() const override
  {
    return m_p->view().size();
  }
  INumericDataInternal* numericData() override { return this; }
  void changeAllocator(const MemoryAllocationOptions& v) override { m_p->changeAllocator(v); }
  void computeHash(DataHashInfo& hash_info) override
  {
    m_p->computeHash(hash_info);
  }

 private:

  ArrayDataT<DataType>* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namesapce Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
