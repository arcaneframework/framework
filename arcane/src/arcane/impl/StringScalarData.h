// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringScalarData.h                                          (C) 2000-2020 */
/*                                                                           */
/* Donnée scalaire de type 'String'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_STRINGSCALARDATA_H
#define ARCANE_IMPL_STRINGSCALARDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

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
 * \brief Donnée scalaire d'une chaîne de caractères unicode.
 */
class StringScalarData
: public ReferenceCounterImpl
, public IScalarDataT<String>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  typedef String DataType;
  typedef StringScalarData ThatClass;
  typedef IScalarDataT<String> DataInterfaceType;

 public:
  StringScalarData(ITraceMng* trace)
  : m_trace(trace) {}
  explicit StringScalarData(const DataStorageBuildInfo& dsbi);
  StringScalarData(const StringScalarData& rhs)
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
  IData* clone() override { return cloneTrue(); }
  IData* cloneEmpty() override { return cloneTrueEmpty(); }
  Ref<IData> cloneRef() override { return makeRef(cloneTrue()); }
  Ref<IData> cloneEmptyRef() override { return makeRef(cloneTrueEmpty()); }
  DataStorageTypeInfo storageTypeInfo() const override;
  StringScalarData* cloneTrue() override { return new ThatClass(*this); }
  StringScalarData* cloneTrueEmpty() override { return new ThatClass(m_trace); }
  Ref<DataInterfaceType> cloneTrueRef() override { DataInterfaceType* d = new ThatClass(*this); return makeRef(d); }
  Ref<DataInterfaceType> cloneTrueEmptyRef() override { DataInterfaceType* d = new ThatClass(m_trace); return makeRef(d); }
  void fillDefault() override
  {
    m_value = String();
  }
  void setName(const String& name) override;
  const ISerializedData* createSerializedData(bool use_basic_type) const override;
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
