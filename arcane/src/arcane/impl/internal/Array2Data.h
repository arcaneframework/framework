// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Data.h                                                (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array2'.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_ARRAY2DATA_H
#define ARCANE_IMPL_INTERNAL_ARRAY2DATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"

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
  Ref<DataInterfaceType> cloneTrueRef() override
  {
    auto* d = _cloneTrue();
    return makeRef(d);
  }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override
  {
    auto* d = _cloneTrueEmpty();
    return makeRef(d);
  }
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

template <typename DataType>
class Array2DataT<DataType>::Impl
: public IArray2DataInternalT<DataType>
, public INumericDataInternal
{
 public:

  explicit Impl(Array2DataT<DataType>* p)
  : m_p(p)
  {}

 public:

  void reserve(Integer new_capacity) override { m_p->m_value.reserve(new_capacity); }
  void resizeOnlyDim1(Int32 new_dim1_size) override
  {
    m_p->m_value.resize(new_dim1_size, m_p->m_value.dim2Size());
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
    compressor->compress(bytes, buf.m_buffer);
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
    m_p->m_value.resize(buf.m_original_dim1_size, buf.m_original_dim2_size);
    Span<DataType> values = m_p->m_value.to1DSpan();
    compressor->decompress(buf.m_buffer, asWritableBytes(values));
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
