// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializedData.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Serialized data.                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISerializedData.h"

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayShape.h"

#include "arcane/core/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  const Int64 SERIALIZE_MAGIC_NUMBER = 0x6b90ac81;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for serialized data.
 */
class SerializedData
: public ReferenceCounterImpl
, public ISerializedData
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  SerializedData();
  SerializedData(eDataType base_data_type, Int64 memory_size,
                 Integer nb_dimension, Int64 nb_element, Int64 nb_base_element,
                 bool is_multi_size, Int64ConstArrayView extents, const ArrayShape& shape);

 public:

  eDataType baseDataType() const override { return m_base_data_type; }
  Integer nbDimension() const override { return m_nb_dimension; }
  Int64 nbElement() const override { return m_nb_element; }
  bool isMultiSize() const override { return m_is_multi_size; }
  Int64 memorySize() const override { return m_memory_size; }
  Int64ConstArrayView extents() const override { return m_extents; }
  ArrayShape shape() const override { return m_shape; }
  Int64 nbBaseElement() const override { return m_nb_base_element; }
  ByteConstArrayView buffer() const override { return m_const_buffer.constSmallView(); }
  ByteArrayView buffer() override { return m_buffer.smallView(); }
  Span<const Byte> bytes() const override { return m_const_buffer; }
  Span<const Byte> constBytes() const override { return m_const_buffer; }
  Span<Byte> bytes() override { return m_buffer; }
  void setBuffer(ByteArrayView buffer) override;
  void setBuffer(ByteConstArrayView buffer) override;
  void setBytes(Span<Byte> bytes) override { setWritableBytes(bytes); }
  void setBytes(Span<const Byte> bytes) override { setConstBytes(bytes); }
  Span<Byte> writableBytes() override { return m_buffer; }
  void setWritableBytes(Span<Byte> bytes) override;
  void setConstBytes(Span<const Byte> bytes) override;
  void allocateMemory(Int64 size) override;

 public:

  void serialize(ISerializer* buffer) override;
  void serialize(ISerializer* buffer) const override;

 public:

  void computeHash(IHashAlgorithm* algo, ByteArray& output) const override;

 private:

  eDataType m_base_data_type;
  Int64 m_memory_size;
  Integer m_nb_dimension;
  Int64 m_nb_element;
  Int64 m_nb_base_element;
  bool m_is_multi_size;
  // TODO: remove the 'm_dimensions' field, but this implies changing
  // the value of computeHash(), so we need to determine the best time
  // to do it.
  UniqueArray<Int32> m_dimensions;
  UniqueArray<Int64> m_extents;
  Int64 m_element_size;
  Span<Byte> m_buffer;
  Span<const Byte> m_const_buffer;
  UniqueArray<Byte> m_stored_buffer;
  ArrayShape m_shape;

 private:

  void _serialize(ISerializer* sbuf) const;
  void _serializeRead(ISerializer* sbuf);
  void _copyExtentsToDimensions();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializedData::
SerializedData()
: m_base_data_type(DT_Unknown)
, m_memory_size(0)
, m_nb_dimension(0)
, m_nb_element(0)
, m_nb_base_element(0)
, m_is_multi_size(false)
, m_element_size(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializedData::
SerializedData(eDataType base_data_type, Int64 memory_size,
               Integer nb_dimension, Int64 nb_element, Int64 nb_base_element,
               bool is_multi_size, Int64ConstArrayView extents, const ArrayShape& shape)
: m_base_data_type(base_data_type)
, m_memory_size(memory_size)
, m_nb_dimension(nb_dimension)
, m_nb_element(nb_element)
, m_nb_base_element(nb_base_element)
, m_is_multi_size(is_multi_size)
, m_extents(extents)
, m_element_size(dataTypeSize(m_base_data_type))
, m_shape(shape)
{
  _copyExtentsToDimensions();
  if (!is_multi_size && shape.nbDimension() == 0) {
    if (nb_dimension == 1) {
      m_shape.setNbDimension(1);
      m_shape.setDimension(0, 1);
    }
    else if (nb_dimension > 1) {
      m_shape.setNbDimension(nb_dimension - 1);
      for (Int32 i = 1; i < nb_dimension; ++i)
        m_shape.setDimension(i - 1, CheckedConvert::toInt32(extents[i]));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
_copyExtentsToDimensions()
{
  Integer n = m_extents.size();
  m_dimensions.resize(n);
  // Exceptions should not be thrown if bounds are exceeded, otherwise
  // the code will throw an exception as soon as the number of array elements
  // exceeds 32 bits. This is not very serious if the values of 'm_dimensions'
  // are invalid because they are only used in computeHash() to
  // maintain compatibility.
  for (Integer i = 0; i < n; ++i)
    m_dimensions[i] = static_cast<Int32>(m_extents[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setBuffer(ByteArrayView buffer)
{
  setBytes(Span<Byte>(buffer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setBuffer(ByteConstArrayView buffer)
{
  setBytes(Span<const Byte>(buffer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setWritableBytes(Span<Byte> buffer)
{
  m_buffer = buffer;
  m_const_buffer = buffer;
  m_stored_buffer.clear();
  m_memory_size = buffer.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setConstBytes(Span<const Byte> buffer)
{
  m_const_buffer = buffer;
  m_buffer = Span<Byte>();
  m_stored_buffer.clear();
  m_memory_size = buffer.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
allocateMemory(Int64 size)
{
  m_stored_buffer = UniqueArray<Byte>(size);
  m_buffer = m_stored_buffer;
  m_const_buffer = m_stored_buffer.view();
  m_memory_size = size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
computeHash(IHashAlgorithm* algo, ByteArray& output) const
{
  // TODO: implement 64-bit support, but this changes the hash.
  algo->computeHash64(m_const_buffer, output);
  const Byte* ptr = reinterpret_cast<const Byte*>(m_dimensions.data());
  Integer msize = CheckedConvert::multiply(m_dimensions.size(), (Integer)sizeof(Integer));
  ByteConstArrayView dim_bytes(msize, ptr);
  algo->computeHash64(dim_bytes, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo Do not use the DT_Byte type for serialization, but * the true base
 * type: this type can be used with MPI, and in this * case, if the machines
 * are heterogeneous, the type information is lost * and the put operation
 * may not match.
 */
void SerializedData::
serialize(ISerializer* sbuf) const
{
  _serialize(sbuf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
serialize(ISerializer* sbuf)
{
  ISerializer::eMode mode = sbuf->mode();

  switch (mode) {
  case ISerializer::ModeReserve:
    _serialize(sbuf);
    break;
  case ISerializer::ModePut:
    _serialize(sbuf);
    break;
  case ISerializer::ModeGet:
    switch (sbuf->readMode()) {
    case ISerializer::ReadReplace:
      _serializeRead(sbuf);
      break;
    case ISerializer::ReadAdd:
      ARCANE_THROW(NotImplementedException, "ReadAdd");
      break;
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
_serializeRead(ISerializer* sbuf)
{
  Int64 magic_number = sbuf->getInt64(); // Magic number for verification
  if (magic_number != SERIALIZE_MAGIC_NUMBER)
    ARCANE_FATAL("Bad magic number for SerializedData '{0}", magic_number);

  Int32 version = sbuf->getInt32(); // For the version number
  if (version != 1)
    ARCANE_FATAL("Bad magic number for SerializedData '{0}", magic_number);

  m_base_data_type = (eDataType)sbuf->getInteger(); // For m_base_data_type
  m_memory_size = sbuf->getInt64(); // For m_memory_size
  m_nb_dimension = sbuf->getInteger(); // For m_nb_dimension
  m_nb_element = sbuf->getInt64(); // For m_nb_element
  m_nb_base_element = sbuf->getInt64(); // For m_nb_base_element
  m_is_multi_size = sbuf->getInteger(); // For m_is_multi_size
  m_element_size = sbuf->getInt64(); // For m_element_size

  // Reading dimensions
  Int64 dimensions_size = sbuf->getInt64();
  m_extents.resize(dimensions_size);
  sbuf->getSpan(m_extents);
  _copyExtentsToDimensions();

  // Reading 'm_shape'
  Int32 shape_nb_dim = sbuf->getInt32(); // For m_shape.nbDimension()
  std::array<Int32, ArrayShape::MAX_NB_DIMENSION> shape_dims_buf;
  Span<Int32> shape_dims(shape_dims_buf.data(), shape_nb_dim);
  sbuf->getSpan(shape_dims); // For the dimensions
  m_shape.setDimensions(shape_dims);

  Int64 buffer_size = sbuf->getInt64();
  m_stored_buffer.resize(buffer_size);
  sbuf->getSpan(m_stored_buffer); // For the data
  m_buffer = m_stored_buffer;
  m_const_buffer = m_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo Do not use the DT_Byte type for serialization, but the true base
 * type: this type can be used with MPI, and in this case, if the machines
 * are heterogeneous, the type information is lost and the put operation may
 * not match.
 */
void SerializedData::
_serialize(ISerializer* sbuf) const
{
  ISerializer::eMode mode = sbuf->mode();
  if (m_extents.size() != m_dimensions.size())
    ARCANE_FATAL("Incoherence between extents ({0}) and dimensions ({1})",
                 m_extents.size(), m_dimensions.size());

  switch (mode) {
  case ISerializer::ModeReserve:
    sbuf->reserveInt64(1); // Magic number for verification
    sbuf->reserveInt32(1); // Version number
    sbuf->reserveInteger(1); // For m_base_data_type
    sbuf->reserveInt64(1); // For m_memory_size
    sbuf->reserveInteger(1); // For m_nb_dimension
    sbuf->reserveInt64(1); // For m_nb_element
    sbuf->reserveInt64(1); // For m_nb_base_element
    sbuf->reserveInteger(1); // For m_is_multi_size
    sbuf->reserveInt64(1); // For m_element_size

    sbuf->reserveInt64(1); // For m_extents.size()
    sbuf->reserveSpan(eBasicDataType::Int64, m_extents.size()); // For the dimensions

    sbuf->reserveInt32(1); // For the number of values in 'm_shape'
    sbuf->reserveSpan(eBasicDataType::Int32, m_shape.nbDimension()); // For 'm_shape' data

    sbuf->reserveInt64(1); // For m_const_buffer.size()
    sbuf->reserveSpan(eBasicDataType::Byte, m_const_buffer.size()); // For the data
    break;
  case ISerializer::ModePut:
    sbuf->putInt64(SERIALIZE_MAGIC_NUMBER); // Magic number for verification
    sbuf->putInt32(1); // Version number
    sbuf->putInteger(m_base_data_type); // For m_base_data_type
    sbuf->putInt64(m_memory_size); // For m_memory_size
    sbuf->putInteger(m_nb_dimension); // For m_nb_dimension
    sbuf->putInt64(m_nb_element); // For m_nb_element
    sbuf->putInt64(m_nb_base_element); // For m_nb_base_element
    sbuf->putInteger(m_is_multi_size); // For m_is_multi_size
    sbuf->putInt64(m_element_size); // For m_element_size

    sbuf->putInt64(m_extents.size()); // For m_extents.size()
    sbuf->putSpan(m_extents); // For the dimensions

    sbuf->putInt32(m_shape.nbDimension()); // For m_shape.nbDimension()
    sbuf->putSpan(m_shape.dimensions()); // For the dimensions

    sbuf->putInt64(m_const_buffer.size()); // For m_const_buffer.size()
    sbuf->putSpan(m_const_buffer); // For the data
    break;
  case ISerializer::ModeGet:
    ARCANE_THROW(NotSupportedException, "ModeGet in const method");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions, const ArrayShape& shape)
{
  return createRef<SerializedData>(data_type, memory_size, nb_dim, nb_element,
                                   nb_base_element, is_multi_size, dimensions, shape);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions)
{
  return arcaneCreateSerializedDataRef(data_type, memory_size, nb_dim, nb_element,
                                       nb_base_element, is_multi_size, dimensions, ArrayShape());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateEmptySerializedDataRef()
{
  return createRef<SerializedData>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
