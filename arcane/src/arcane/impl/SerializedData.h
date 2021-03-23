// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializedData.h                                            (C) 2000-2020 */
/*                                                                           */
/* Donnée sérialisée.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_SERIALIZEDDATA_H
#define ARCANE_IMPL_SERIALIZEDDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/ISerializedData.h"

#include "arcane/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée sérialisée.
 */
class SerializedData
: public ReferenceCounterImpl
, public ISerializedData
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:
  
  SerializedData();
  ARCANE_DEPRECATED_2018_R("Use constructor using extents")
  SerializedData(eDataType base_data_type,Int64 memory_size,
                 Integer nb_dimension,Int64 nb_element,Int64 nb_base_element,
                 bool is_multi_size,Int32ConstArrayView dimensions);
  SerializedData(eDataType base_data_type,Int64 memory_size,
                 Integer nb_dimension,Int64 nb_element,Int64 nb_base_element,
                 bool is_multi_size,Int64ConstArrayView extents);

 public:

  eDataType baseDataType() const override { return m_base_data_type; }
  Integer nbDimension() const override { return m_nb_dimension; }
  Int64 nbElement() const override { return m_nb_element; }
  bool isMultiSize() const override { return m_is_multi_size; }
  Int64 memorySize() const override { return m_memory_size; }
  IntegerConstArrayView dimensions() const override { return m_dimensions; }
  Int64ConstArrayView extents() const override { return m_extents; }
  Int64 nbBaseElement() const override { return m_nb_base_element; }
  ByteConstArrayView buffer() const override { return m_const_buffer.constSmallView(); }
  ByteArrayView buffer() override { return m_buffer.smallView(); }
  Span<const Byte> bytes() const override { return m_const_buffer; }
  Span<Byte> bytes() override { return m_buffer; }
  void setBuffer(ByteArrayView buffer) override;
  void setBuffer(ByteConstArrayView buffer) override;
  void setBytes(Span<Byte> bytes) override;
  void setBytes(Span<const Byte> bytes) override;
  void setBuffer(SharedArray<Byte>& buffer) override;
  void allocateMemory(Int64 size) override;

 public:
  
  void serialize(ISerializer* buffer) override;
  void serialize(ISerializer* buffer) const override;

 public:
  
  void computeHash(IHashAlgorithm* algo,ByteArray& output) const override;

 private:
  
  eDataType m_base_data_type;
  Int64 m_memory_size;
  Integer m_nb_dimension;
  Int64 m_nb_element;
  Int64 m_nb_base_element;
  bool m_is_multi_size;
  UniqueArray<Int32> m_dimensions;
  UniqueArray<Int64> m_extents;
  Int64 m_element_size;
  Span<Byte> m_buffer;
  Span<const Byte> m_const_buffer;
  // Une fois que les méthodes obsolètes setBuffer() auront été supprimées on
  // pourra transformer cela en UniqueArray.
  SharedArray<Byte> m_stored_buffer;

 private:
  
  void _serialize(ISerializer* sbuf) const;
  void _copyDimensionsToExtents();
  void _copyExtentsToDimensions();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

