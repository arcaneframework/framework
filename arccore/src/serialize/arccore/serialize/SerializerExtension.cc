// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializerExtension.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Extension of the serializer with support of multidimensional arrays.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializerExtension.h"

#include "arccore/base/Float128.h"
#include "arccore/base/Float16.h"
#include "arccore/base/Int128.h"
#include "arccore/collections/Array2.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/MDSpan.h"
#include "arccore/common/NumArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
reserveSpan(Span2<const Type> values)
{
  Span<Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->reserveSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
reserveSpan(MDSpan<Type, Extents> values)
{
  m_serializer->reserveSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
reserveArray(Array2<const Type> values)
{
  m_serializer->reserveInt64(2);
  m_serializer->reserveSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
reserveArray(NumArray<Type, Extents> values)
{
  constexpr Int32 rank = Extents::rank();
  m_serializer->reserveInt64(rank);
  m_serializer->reserveSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
reserveArray(Span<const String> values) const
{
  // size of each String + size of array.
  m_serializer->reserveInt64(values.size() + 1);
  for (const String& elem : values) {
    m_serializer->reserveSpan(elem.bytes());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
putSpan(Span2<const Type> values)
{
  Span<Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->putSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
putSpan(MDSpan<Type, Extents> values)
{
  m_serializer->putSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
putArray(Array2<const Type> values)
{
  m_serializer->putInt64(values.dim1Size());
  m_serializer->putInt64(values.dim2Size());
  m_serializer->putSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
putArray(NumArray<Type, Extents> values)
{
  constexpr Int32 rank = Extents::rank();
  if constexpr (rank >= 1) {
    m_serializer->putInt64(values.extent0());
  }
  if constexpr (rank >= 2) {
    m_serializer->putInt64(values.extent1());
  }
  if constexpr (rank >= 3) {
    m_serializer->putInt64(values.extent2());
  }
  if constexpr (rank >= 4) {
    m_serializer->putInt64(values.extent3());
  }
  m_serializer->putSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
putArray(Span<const String> values) const
{
  m_serializer->putInt64(values.size());
  for (const String& elem : values) {
    m_serializer->putInt64(elem.length());
    m_serializer->putSpan(elem.bytes());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
getSpan(Span2<Type> values)
{
  Span<Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->getSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
getSpan(MDSpan<Type, Extents> values)
{
  Span<Type> values_1d = values.to1DSpan();
  m_serializer->getSpan(values_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
getArray(Array2<Type>& values)
{
  Int64 size1 = m_serializer->getInt64();
  Int64 size2 = m_serializer->getInt64();

  values.resize(size1, size2);
  m_serializer->getSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
getArray(NumArray<Type, Extents>& values)
{
  using IndexType = Extents::ExtentIndexType;
  constexpr Int32 rank = Extents::rank();
  if constexpr (rank == 1) {
    IndexType size1 = static_cast<IndexType>(m_serializer->getInt64());
    MDIndex<1, IndexType> mdi(size1);
    values.resizeDestructive(mdi);
  }
  if constexpr (rank == 2) {
    IndexType size1 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size2 = static_cast<IndexType>(m_serializer->getInt64());
    MDIndex<2, IndexType> mdi(size1, size2);
    values.resizeDestructive(mdi);
  }
  if constexpr (rank == 3) {
    IndexType size1 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size2 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size3 = static_cast<IndexType>(m_serializer->getInt64());
    MDIndex<3, IndexType> mdi(size1, size2, size3);
    values.resizeDestructive(mdi);
  }
  if constexpr (rank == 4) {
    IndexType size1 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size2 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size3 = static_cast<IndexType>(m_serializer->getInt64());
    IndexType size4 = static_cast<IndexType>(m_serializer->getInt64());
    MDIndex<4, IndexType> mdi(size1, size2, size3, size4);
    values.resizeDestructive(mdi);
  }
  m_serializer->getSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
getArray(Array<String>& values) const
{
  const Int64 size_array = m_serializer->getInt64();
  values.resize(size_array);
  for (Int64 i = 0; i < size_array; ++i) {
    const Int64 len = m_serializer->getInt64();
    UniqueArray<Byte> bytes(len);
    m_serializer->getSpan(bytes);
    values[i] = String(bytes.span());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(return_type, class_method_name, is_const, arg_with_template, et) \
  template return_type class_method_name<is_const Byte>(arg_with_template<is_const Byte> et); \
  template return_type class_method_name<is_const Real>(arg_with_template<is_const Real> et); \
  template return_type class_method_name<is_const Int16>(arg_with_template<is_const Int16> et); \
  template return_type class_method_name<is_const Int32>(arg_with_template<is_const Int32> et); \
  template return_type class_method_name<is_const Int64>(arg_with_template<is_const Int64> et); \
  template return_type class_method_name<is_const Float32>(arg_with_template<is_const Float32> et); \
  template return_type class_method_name<is_const Float16>(arg_with_template<is_const Float16> et); \
  template return_type class_method_name<is_const BFloat16>(arg_with_template<is_const BFloat16> et); \
  template return_type class_method_name<is_const Int8>(arg_with_template<is_const Int8> et); \
  template return_type class_method_name<is_const Float128>(arg_with_template<is_const Float128> et); \
  template return_type class_method_name<is_const Int128>(arg_with_template<is_const Int128> et)

#define ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER2(return_type, class_method_name, is_const, second_template, arg_with_template, et) \
  template return_type class_method_name<is_const Byte, second_template>(arg_with_template<is_const Byte, second_template> et); \
  template return_type class_method_name<is_const Real, second_template>(arg_with_template<is_const Real, second_template> et); \
  template return_type class_method_name<is_const Int16, second_template>(arg_with_template<is_const Int16, second_template> et); \
  template return_type class_method_name<is_const Int32, second_template>(arg_with_template<is_const Int32, second_template> et); \
  template return_type class_method_name<is_const Int64, second_template>(arg_with_template<is_const Int64, second_template> et); \
  template return_type class_method_name<is_const Float32, second_template>(arg_with_template<is_const Float32, second_template> et); \
  template return_type class_method_name<is_const Float16, second_template>(arg_with_template<is_const Float16, second_template> et); \
  template return_type class_method_name<is_const BFloat16, second_template>(arg_with_template<is_const BFloat16, second_template> et); \
  template return_type class_method_name<is_const Int8, second_template>(arg_with_template<is_const Int8, second_template> et); \
  template return_type class_method_name<is_const Float128, second_template>(arg_with_template<is_const Float128, second_template> et); \
  template return_type class_method_name<is_const Int128, second_template>(arg_with_template<is_const Int128, second_template> et)

#define ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(return_type, class_method_name, is_const, arg_with_template, et) \
  ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER2(return_type, class_method_name, is_const, MDDim1, arg_with_template, et); \
  ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER2(return_type, class_method_name, is_const, MDDim2, arg_with_template, et); \
  ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER2(return_type, class_method_name, is_const, MDDim3, arg_with_template, et); \
  ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER2(return_type, class_method_name, is_const, MDDim4, arg_with_template, et)

ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::reserveSpan, const, Span2, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::reserveArray, const, Array2, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::putSpan, const, Span2, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::putArray, const, Array2, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::getSpan, , Span2, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER(void, SerializerExtension::getArray, , Array2, &);

ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::reserveSpan, const, MDSpan, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::reserveArray, , NumArray, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::putSpan, const, MDSpan, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::putArray, , NumArray, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::getSpan, , MDSpan, );
ARCCORE_INTERNAL_INSTANTIATE_MDSERIALIZER3(void, SerializerExtension::getArray, , NumArray, &);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
