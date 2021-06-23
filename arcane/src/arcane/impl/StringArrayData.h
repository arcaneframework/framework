// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringArrayData.h                                           (C) 2000-2021 */
/*                                                                           */
/* Donnée de type 'StringUniqueArray'.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_STRINGARRAYDATA_H
#define ARCANE_IMPL_STRINGARRAYDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/IData.h"
#include "arcane/IDataVisitor.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

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
, public IArrayDataInternalT<String>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  typedef String DataType;
  typedef StringArrayData ThatClass;
  typedef IArrayDataT<String> DataInterfaceType;

 public:
  explicit StringArrayData(ITraceMng* trace)
  : m_trace(trace) {}
  explicit StringArrayData(const DataStorageBuildInfo& dsbi);
  StringArrayData(const StringArrayData& rhs)
  : m_value(rhs.m_value)
  , m_trace(rhs.m_trace)
  {}

 public:
  Integer dimension() const override { return 1; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  Array<DataType>& value() override { return m_value; }
  const Array<DataType>& value() const override { return m_value; }
  Array<DataType>& _internalDeprecatedValue() override { return m_value; }
  ConstArrayView<DataType> view() const override { return m_value; }
  ArrayView<DataType> view() override { return m_value; }
  void resize(Integer new_size) override { m_value.resize(new_size); }
  void reserve(Integer new_capacity) override { m_value.reserve(new_capacity); }
  IData* clone() override { return cloneTrue(); }
  IData* cloneEmpty() override { return cloneTrueEmpty(); }
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataInterfaceType* cloneTrue() override { return _cloneTrue(); }
  DataInterfaceType* cloneTrueEmpty() override { return _cloneTrueEmpty(); }
  Ref<DataInterfaceType> cloneTrueRef() override { auto* d = _cloneTrue(); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { auto* d = _cloneTrueEmpty(); return makeRef(d); }
  DataStorageTypeInfo storageTypeInfo() const override;
  void fillDefault() override { m_value.fill(String()); }
  void setName(const String& name) override;
  const ISerializedData* createSerializedData(bool use_basic_type) const override;
  Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const override;
  void allocateBufferForSerializedData(ISerializedData* sdata) override;
  void assignSerializedData(const ISerializedData* sdata) override;
  void copy(const IData* data) override;
  void swapValues(IData* data) override;
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
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
  void visitMultiArray2(IMultiArray2DataVisitor* visitor) override;

 public:

  IArrayDataInternalT<DataType>* _internal() override { return this; }

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  UniqueArray<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  ThatClass* _cloneTrue() const { return new ThatClass(*this); }
  ThatClass* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
