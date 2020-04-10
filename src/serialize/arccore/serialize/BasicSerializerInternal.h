// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerializerInternal.h                                   (C) 2000-2020 */
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

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicSerializer::Impl
{
 public:
  virtual ~Impl(){}
 public:
  virtual Span<Real> getRealBuffer() =0;
  virtual Span<Int16> getInt16Buffer() =0;
  virtual Span<Int32> getInt32Buffer() =0;
  virtual Span<Int64> getInt64Buffer() =0;
  virtual Span<Byte> getByteBuffer() =0;
  virtual void allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
                              Int64 nb_int64,Int64 nb_byte) =0;
  virtual void copy(Impl* rhs) =0;
  virtual Span<Byte> globalBuffer() =0;
  virtual Span<const Byte> globalBuffer() const =0;
  virtual Int64ConstArrayView sizesBuffer() const =0;
  virtual ByteConstArrayView copyAndGetSizesBuffer() =0;
  virtual void preallocate(Int64 size) =0;
  virtual void setFromSizes() =0;
  virtual Int64 totalSize() const =0;
  virtual void printSizes(std::ostream& o) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicSerializer::Impl2
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
 public:
  void reserve(eDataType dt,Int64 n,Int64 nb_put);
  void putType(eBasicDataType t);
  void getAndCheckType(eBasicDataType expected_type);
  void allocateBuffer();
  void allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
                      Int64 nb_int64,Int64 nb_byte);
  void copy(const BasicSerializer& rhs);
  void setMode(eMode new_mode);
  void setFromSizes();
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
 private:
  void _setViews();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
