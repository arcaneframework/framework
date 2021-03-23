// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Data.h                                                (C) 2000-2020 */
/*                                                                           */
/* Donnée du type 'Array2'.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ARRAY2DATA_H
#define ARCANE_IMPL_ARRAY2DATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"
#include "arcane/utils/NotSupportedException.h"

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
 * \brief Donnée tableau bi-dimensionnel d'un type \a DataType
 */
template <class DataType>
class Array2DataT
: public ReferenceCounterImpl
, public IArray2DataT<DataType>
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  typedef Array2DataT<DataType> ThatClass;
  typedef IArray2DataT<DataType> DataInterfaceType;

 public:

  explicit Array2DataT(ITraceMng* trace);
  explicit Array2DataT(const DataStorageBuildInfo& dsbi);
  Array2DataT(const Array2DataT<DataType>& rhs);

 public:
  Integer dimension() const override { return 2; }
  Integer multiTag() const override { return 0; }
  eDataType dataType() const override { return DataTypeTraitsT<DataType>::type(); }
  void serialize(ISerializer* sbuf, IDataOperation* operation) override;
  void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) override;
  Array2<DataType>& value() override { return m_value; }
  const Array2<DataType>& value() const override { return m_value; }
  void resize(Integer new_size) override;
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
  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;
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
  void visitMultiArray2(IMultiArray2DataVisitor*) override
  {
    ARCANE_THROW(NotSupportedException, "Can not visit multiarray2 data with array data");
  }

 public:

  void swapValuesDirect(ThatClass* true_data);

 public:

  static DataStorageTypeInfo staticStorageTypeInfo();

 private:

  UniqueArray2<DataType> m_value; //!< Donnée
  ITraceMng* m_trace;

 private:

  IArray2DataT<DataType>* _cloneTrue() const { return new ThatClass(*this); }
  IArray2DataT<DataType>* _cloneTrueEmpty() const { return new ThatClass(m_trace); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
