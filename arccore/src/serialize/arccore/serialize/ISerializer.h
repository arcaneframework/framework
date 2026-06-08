// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializer.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a serializer.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_ISERIALIZER_H
#define ARCCORE_SERIALIZE_ISERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializeGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/base/RefDeclarations.h"
#include "arccore/base/BasicDataType.h"
#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Serializer interface.
 *
 * It is possible to create an instance of this class via the method
 * createSerializer();
 *
 * This interface manages a serializer to store and read a set of values.
 * Serialization takes place in three phases:
 *
 * 1. each object calls one or more of the methods reserve()/reserveSpan()
 *    to indicate how much memory it needs</li>
 * 2. the memory is allocated by allocateBuffer()</li>
 * 3. each object calls one or more of the methods put()/putSpan() to add
 *    its information to the serializer</li>
 *
 * Deserialization is done identically but uses the functions get()/getSpan().
 * The operation is similar to a queue: for every get()/getSpan(), a previous
 * put()/putSpan() must correspond, and the get()/getSpan() and the put()/putSpan()
 * must be in the same order.
 *
 * It is possible to use overloads of reserve()/get()/put(). In this case, you
 * must ensure consistency in their usage. For example, if you call reserveSpan(),
 * you must then call putSpan() and getSpan().
 *
 * \todo add example.
 */
class ARCCORE_SERIALIZE_EXPORT ISerializer
{
 public:

  //! Serializer operating mode
  enum eMode
  {
    ModeReserve, //! The serializer expects reserve()
    ModePut, //!< The serializer expects put()
    ModeGet //!< The serializer expects get()
  };
  //! Serializer read mode
  enum eReadMode
  {
    ReadReplace, //!< Replace current elements with those read
    ReadAdd //!< Add those read to the current elements
  };

  // NOTE: do not change these values as they are used in Arcane
  enum eDataType
  {
    DT_Byte = 0, //!< Byte data type
    DT_Real = 1, //!< Real data type
    DT_Int16 = 2, //!< 16-bit integer data type
    DT_Int32 = 3, //!< 32-bit integer data type
    DT_Int64 = 4, //!< 64-bit integer data type
    DT_Float32 = 12, //!< 32-bit floating point data type
    DT_Float16 = 11, //!< 16-bit floating point data type
    DT_BFloat16 = 10, //!< 'brain float' data type
    DT_Int8 = 13, //!< 8-bit integer data type
    DT_Float128 = 14, //!< 128-bit floating point data type
    DT_Int128 = 15, //!< 128-bit integer data type
    DT_Float64 = DT_Real
  };

  virtual ~ISerializer() = default; //!< Frees resources

 public:

  /*!
   * \brief Reserves memory for \a n values of \a dt.
   *
   * A call to a putSpan() method must be made for the
   * serialization to be correct.
   */
  virtual void reserveSpan(eBasicDataType dt, Int64 n) = 0;

  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Real> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Int16> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Int32> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Int64> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Byte> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Int8> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Float16> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const BFloat16> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Float32> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Float128> values);
  //! Reserve for a view of \a values elements
  virtual void reserveSpan(Span<const Int128> values);

  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Real> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Int16> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Int32> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Int64> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Byte> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Int8> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Float16> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Float32> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const BFloat16> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Float128> values) = 0;
  //! Reserve to save the number of elements and the \a values elements
  virtual void reserveArray(Span<const Int128> values) = 0;

  /*!
   * \brief Reserves memory for \a n objects of type \a dt.
   *
   * \a n calls to a put() method with a single value must be made for
   * the serialization to be correct.
   *
   * If you want to serialize multiple values with a single call to put(),
   * you must use the reserveSpan() method.
   */
  virtual void reserve(eBasicDataType dt, Int64 n) = 0;

  virtual void reserveInteger(Int64 n) = 0;

  //! Reserve memory for a character string \a str.
  virtual void reserve(const String& str) = 0;

  //! Reserve for \a n Real
  void reserveReal(Int64 n) { reserve(eBasicDataType::Real, n); }
  //! Reserve for \a n Int16
  void reserveInt16(Int64 n) { reserve(eBasicDataType::Int16, n); }
  //! Reserve for \a n Int64
  void reserveInt64(Int64 n) { reserve(eBasicDataType::Int64, n); }
  //! Reserve for \a n Int32
  void reserveInt32(Int64 n) { reserve(eBasicDataType::Int32, n); }
  //! Reserve for \a n Byte
  void reserveByte(Int64 n) { reserve(eBasicDataType::Byte, n); }
  //! Reserve for \a n Int8
  void reserveInt8(Int64 n) { reserve(eBasicDataType::Int8, n); }
  //! Reserve for \a n Float16
  void reserveFloat16(Int64 n) { reserve(eBasicDataType::Float16, n); }
  //! Reserve for \a n Float32
  void reserveFloat32(Int64 n) { reserve(eBasicDataType::Float32, n); }
  //! Reserve for \a n BFloat16
  void reserveBFloat16(Int64 n) { reserve(eBasicDataType::BFloat16, n); }
  //! Reserve for \a n Float128
  void reserveFloat128(Int64 n) { reserve(eBasicDataType::Float128, n); }
  //! Reserve for \a n Int128
  void reserveInt128(Int64 n) { reserve(eBasicDataType::Int128, n); }

 public:

  /*!
   * \brief Reserves memory for \a n values of \a dt.
   *
   * \dt must be an integral type: DT_Int16, DT_Int32, DT_Int64,
   * DT_Real or DT_Byte.
   *
   * A call to a putSpan() method must be made for the serialization to be correct.
   *
   * \deprecated Use reserveSpan(eBasicDataType) instead
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use reserveSpan(eBasicDataType) instead")
  virtual void reserveSpan(eDataType dt, Int64 n) = 0;

  //! \sa reserve(eDataType dt,Int64 n)
  ARCCORE_DEPRECATED_REASON("Y2024: Use reserveSpan(eBasicDataType) instead")
  void reserveSpan(int dt, Int64 n);
  /*!
   * \brief Reserves memory for \a n objects of type \a dt.
   *
   * \dt must be an integral type: DT_Int16, DT_Int32, DT_Int64,
   * DT_Real or DT_Byte.
   *
   * \a n calls to a put() method with a single value must be made for the
   * serialization to be correct.
   *
   * If you want to serialize multiple values with a single call to put(),
   * you must use the reserveSpan() method.
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use reserveSpan(eBasicDataType) instead")
  virtual void reserve(eDataType dt, Int64 n) = 0;

  //! \sa reserve(eDataType dt,Int64 n)
  ARCCORE_DEPRECATED_REASON("Y2024: Use reserveSpan(eBasicDataType) instead")
  void reserve(int dt, Int64 n);

 public:

  //! Add the array \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Real> values) = 0;
  //! Add the array \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int16> values) = 0;
  //! Add the array \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int32> values) = 0;
  //! Add the array \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int64> values) = 0;
  //! Add the array \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Byte> values) = 0;

 public:

  //! Add the string \a value
  virtual void put(const String& value) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const Real> values);
  //! Add the array \a values
  virtual void putSpan(Span<const Int16> values);
  //! Add the array \a values
  virtual void putSpan(Span<const Int32> values);
  //! Add the array \a values
  virtual void putSpan(Span<const Int64> values);
  //! Add the array \a values
  virtual void putSpan(Span<const Byte> values);
  //! Add the array \a values
  virtual void putSpan(Span<const Int8> values) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const Float16> values) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const BFloat16> values) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const Float32> values) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const Float128> values) = 0;
  //! Add the array \a values
  virtual void putSpan(Span<const Int128> values) = 0;

  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Real> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Int16> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Int32> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Int64> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Byte> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Int8> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Float16> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const BFloat16> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Float32> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Float128> values) = 0;
  //! Save the number of elements and the \a values elements
  virtual void putArray(Span<const Int128> values) = 0;

  //! Add \a value
  virtual void put(Real value) = 0;
  //! Add \a value
  virtual void put(Int16 value) = 0;
  //! Add \a value
  virtual void put(Int32 value) = 0;
  //! Add \a value
  virtual void put(Int64 value) = 0;
  //! Add value
  virtual void put(Byte value) = 0;
  //! Add value
  virtual void put(Int8 value) = 0;
  //! Add value
  virtual void put(Float16 value) = 0;
  //! Add value
  virtual void put(BFloat16 value) = 0;
  //! Add value
  virtual void put(Float32 value) = 0;
  //! Add value
  virtual void put(Float128 value) = 0;
  //! Add value
  virtual void put(Int128 value) = 0;

  //! Add the real \a value
  virtual void putReal(Real value) = 0;
  //! Add the integer \a value
  virtual void putInt16(Int16 value) = 0;
  //! Add the integer \a value
  virtual void putInt32(Int32 value) = 0;
  //! Add the integer \a value
  virtual void putInt64(Int64 value) = 0;
  //! Add the integer \a value
  virtual void putInteger(Integer value) = 0;
  //! Add the byte \a value
  virtual void putByte(Byte value) = 0;
  //! Add \a value
  virtual void putInt8(Int8 value) = 0;
  //! Add \a value
  virtual void putFloat16(Float16 value) = 0;
  //! Add \a value
  virtual void putBFloat16(BFloat16 value) = 0;
  //! Add \a value
  virtual void putFloat32(Float32 value) = 0;
  //! Add \a value
  virtual void putFloat128(Float128 value) = 0;
  //! Add \a value
  virtual void putInt128(Int128 value) = 0;

 public:

  //! Retrieve the array \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ArrayView<Real> values) = 0;
  //! Retrieve the array \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ArrayView<Int16> values) = 0;
  //! Retrieve the array \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ArrayView<Int32> values) = 0;
  //! Retrieve the array \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ArrayView<Int64> values) = 0;
  //! Retrieve the array \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ArrayView<Byte> values) = 0;

 public:

  //! Retrieve the string \a value
  virtual void get(String& value) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<Real> values);
  //! Retrieve the array \a values
  virtual void getSpan(Span<Int16> values);
  //! Retrieve the array \a values
  virtual void getSpan(Span<Int32> values);
  //! Retrieve the array \a values
  virtual void getSpan(Span<Int64> values);
  //! Retrieve the array \a values
  virtual void getSpan(Span<Byte> values);
  //! Retrieve the array \a values
  virtual void getSpan(Span<Int8> values) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<Float16> values) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<BFloat16> values) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<Float32> values) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<Float128> values) = 0;
  //! Retrieve the array \a values
  virtual void getSpan(Span<Int128> values) = 0;

  //! Resize and fill \a values
  virtual void getArray(Array<Real>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Int16>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Int32>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Int64>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Byte>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Int8>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Float16>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<BFloat16>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Float32>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Float128>& values) = 0;
  //! Resize and fill \a values
  virtual void getArray(Array<Int128>& values) = 0;

  //! Retrieve a real number
  virtual Real getReal() = 0;
  //! Retrieve a 16-bit integer
  virtual Int16 getInt16() = 0;
  //! Retrieve an integer
  virtual Int32 getInt32() = 0;
  //! Retrieve a size
  virtual Int64 getInt64() = 0;
  //! Retrieve a size
  virtual Integer getInteger() = 0;
  //! Retrieve a byte
  virtual Byte getByte() = 0;
  //! Retrieve an Int8
  virtual Int8 getInt8() = 0;
  //! Retrieve a Float16
  virtual Float16 getFloat16() = 0;
  //! Retrieve a BFloat16
  virtual BFloat16 getBFloat16() = 0;
  //! Retrieve a Float32
  virtual Float32 getFloat32() = 0;
  //! Retrieve a Float128
  virtual Float128 getFloat128() = 0;
  //! Retrieve an Int128
  virtual Int128 getInt128() = 0;

  //! Allocates the serializer memory
  virtual void allocateBuffer() = 0;

  ARCCORE_DEPRECATED_2020("Internal method. Do not use")
  virtual void allocateBuffer(Int64 nb_real, Int64 nb_int16, Int64 nb_int32,
                              Int64 nb_int64, Int64 nb_byte) = 0;

  //! Current operating mode
  virtual eMode mode() const = 0;
  //! Sets the current mode
  virtual void setMode(eMode new_mode) = 0;

  //! Read mode
  virtual eReadMode readMode() const = 0;
  //! Sets the read mode
  virtual void setReadMode(eReadMode read_mode) = 0;

  //! Copies the data from \a from into this instance
  virtual void copy(const ISerializer* from) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates an instance of ISerializer
 */
extern "C++" ARCCORE_SERIALIZE_EXPORT Ref<ISerializer>
createSerializer();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::createSerializer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
