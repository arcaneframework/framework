// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializerInternal.h                                   (C) 2000-2025 */
/*                                                                           */
/* Partie interne de 'BasicSerializer'.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_BASICSERIALIZERINTERNAL_H
#define ARCCORE_SERIALIZE_BASICSERIALIZERINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"
#include "arccore/base/BasicDataType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_SERIALIZE_EXPORT BasicSerializer::Impl
{
 public:

  virtual ~Impl() {}

 public:

  virtual Span<Real> getRealBuffer() = 0;
  virtual Span<Int16> getInt16Buffer() = 0;
  virtual Span<Int32> getInt32Buffer() = 0;
  virtual Span<Int64> getInt64Buffer() = 0;
  virtual Span<Byte> getByteBuffer() = 0;
  virtual Span<Int8> getInt8Buffer() = 0;
  virtual Span<Float16> getFloat16Buffer() = 0;
  virtual Span<BFloat16> getBFloat16Buffer() = 0;
  virtual Span<Float32> getFloat32Buffer() = 0;
  virtual Span<Float128> getFloat128Buffer() = 0;
  virtual Span<Int128> getInt128Buffer() = 0;

  virtual void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                              Int64 nb_int64, Int64 nb_byte, Int64 nb_int8, Int64 nb_float16,
                              Int64 nb_bfloat16, Int64 nb_float32, Int64 nb_float128, Int64 nb_int128) = 0;
  virtual void copy(Impl* rhs) = 0;
  virtual Span<Byte> globalBuffer() = 0;
  virtual Span<const Byte> globalBuffer() const = 0;
  virtual ConstArrayView<Int64> sizesBuffer() const = 0;
  virtual ConstArrayView<Byte> copyAndGetSizesBuffer() = 0;
  virtual void preallocate(Int64 size) = 0;
  virtual void releaseBuffer() = 0;
  virtual void setFromSizes() = 0;
  virtual Int64 totalSize() const = 0;
  virtual void printSizes(std::ostream& o) const = 0;

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: use overload with float16/float32")
  virtual void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                              Int64 nb_int64, Int64 nb_byte) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_SERIALIZE_EXPORT BasicSerializer::Impl2
{
 public:

  Impl2();
  ~Impl2();

 public:

  Span<const Real> realBytes() const { return m_real.m_buffer; }
  Span<const Int64> int64Bytes() const { return m_int64.m_buffer; }
  Span<const Int32> int32Bytes() const { return m_int32.m_buffer; }
  Span<const Int16> int16Bytes() const { return m_int16.m_buffer; }
  Span<const Byte> byteBytes() const { return m_byte.m_buffer; }
  Span<const Int8> int8Bytes() const { return m_int8.m_buffer; }
  Span<const Float16> float16Bytes() const { return m_float16.m_buffer; }
  Span<const BFloat16> bfloat16Bytes() const { return m_bfloat16.m_buffer; }
  Span<const Float32> float32Bytes() const { return m_float32.m_buffer; }
  Span<const Float128> float128Bytes() const { return m_float128.m_buffer; }
  Span<const Int128> int128Bytes() const { return m_int128.m_buffer; }

 public:

  ARCCORE_DEPRECATED_REASON("Y2024: Use reserve(eBasicDataType) instead")
  void reserve(eDataType dt, Int64 n, Int64 nb_put);
  void reserve(eBasicDataType dt, Int64 n, Int64 nb_put);
  void putType(eBasicDataType t);
  void getAndCheckType(eBasicDataType expected_type);
  void allocateBuffer();
  void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                      Int64 nb_int64, Int64 nb_byte, Int64 nb_int8, Int64 nb_float16,
                      Int64 nb_bfloat16, Int64 nb_float32, Int64 nb_float128, Int64 nb_int128);
  void copy(const BasicSerializer& rhs);
  void setMode(eMode new_mode);
  void setFromSizes();

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: use overload with float16/float32")
  void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                      Int64 nb_int64, Int64 nb_byte);

 public:

  void setSerializeTypeInfo(bool v) { m_is_serialize_typeinfo = v; }
  bool isSerializeTypeInfo() const { return m_is_serialize_typeinfo; }

 public:

  eMode m_mode;
  eReadMode m_read_mode;
  BasicSerializer::Impl* m_p;
  bool m_is_serialize_typeinfo = false;
  BasicSerializerDataT<Real> m_real;
  BasicSerializerDataT<Int64> m_int64;
  BasicSerializerDataT<Int32> m_int32;
  BasicSerializerDataT<Int16> m_int16;
  BasicSerializerDataT<Byte> m_byte;
  BasicSerializerDataT<Int8> m_int8;
  BasicSerializerDataT<Float16> m_float16;
  BasicSerializerDataT<BFloat16> m_bfloat16;
  BasicSerializerDataT<Float32> m_float32;
  BasicSerializerDataT<Float128> m_float128;
  BasicSerializerDataT<Int128> m_int128;

 private:

  void _setViews();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
