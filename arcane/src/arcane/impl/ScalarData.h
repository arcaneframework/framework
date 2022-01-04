// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScalarData.h                                                (C) 2000-2020 */
/*                                                                           */
/* Donnée scalaire.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SCALARDATA_H
#define ARCANE_IMPL_SCALARDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/IData.h"
#include "arcane/IDataVisitor.h"

#include "arccore/base/ReferenceCounterImpl.h"

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

  explicit ScalarDataT(ITraceMng* trace)
  : m_value(DataTypeTraitsT<DataType>::defaultValue())
  , m_trace(trace)
  {}
  explicit ScalarDataT(const DataStorageBuildInfo& dsbi);
  ScalarDataT(const ScalarDataT<DataType>& rhs)
  : m_value(rhs.m_value)
  , m_trace(rhs.m_trace)
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
  void visitMultiArray2(IMultiArray2DataVisitor* visitor) override;

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  DataType m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  DataInterfaceType* _cloneTrue() const { return new ThatClass(*this); }
  DataInterfaceType* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
