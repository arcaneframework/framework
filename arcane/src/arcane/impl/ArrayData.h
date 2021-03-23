// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayData.h                                                 (C) 2000-2020 */
/*                                                                           */
/* Donnée de type 'Array'.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ARRAYDATA_H
#define ARCANE_IMPL_ARRAYDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/IData.h"
#include "arcane/IDataVisitor.h"

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Ref.h"

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
 public:

  typedef ArrayDataT<DataType> ThatClass;
  typedef IArrayDataT<DataType> DataInterfaceType;

 public:

  explicit ArrayDataT(ITraceMng* trace);
  explicit ArrayDataT(const DataStorageBuildInfo& dsbi);
  ArrayDataT(const ArrayDataT<DataType>& rhs);

 public:

  Integer dimension() const override { return 1; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf,IDataOperation* operation) override;
  void serialize(ISerializer* sbuf,Int32ConstArrayView ids,IDataOperation* operation) override;
  Array<DataType>& value() override { return m_value; }
  const Array<DataType>& value() const override { return m_value; }
  void resize(Integer new_size) override { m_value.resize(new_size); }
  IData* clone() override { return cloneTrue(); }
  IData* cloneEmpty() override { return cloneTrueEmpty(); }
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
  DataInterfaceType* cloneTrue() override { return _cloneTrue(); }
  DataInterfaceType* cloneTrueEmpty() override { return _cloneTrueEmpty(); }
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
  void computeHash(IHashAlgorithm* algo,ByteArray& output) const override;
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
  void visitMultiArray2(IMultiArray2DataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException,"Can not visit multiarray2 data with array data");
  }

 public:

  void swapValuesDirect(ThatClass* true_data);

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  UniqueArray<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  void _serialize(ISerializer* sbuf,Span<const Int32> ids,IDataOperation* operation);
  IArrayDataT<DataType>* _cloneTrue() const { return new ThatClass(*this); }
  IArrayDataT<DataType>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
