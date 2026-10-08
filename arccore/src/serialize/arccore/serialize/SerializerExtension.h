// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializerExtension.h                                       (C) 2000-2026 */
/*                                                                           */
/* Extension of the serializer with support of multidimensional arrays.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_SERIALIZEREXTENSION_H
#define ARCCORE_SERIALIZE_SERIALIZEREXTENSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/ISerializer.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_SERIALIZE_EXPORT SerializerExtension
{
 public:

  SerializerExtension(ISerializer* serializer)
  : m_serializer(serializer)
  {}

 public:

  template <class Type>
  void reserveSpan(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void reserveSpan(MDSpan<const Type, Extents> values) const;

  template <class Type>
  void reserveArray(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void reserveArray(MDSpan<const Type, Extents> values) const;

  void reserveArray(Span<const String> values) const;

 public:

  template <class Type>
  void putSpan(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void putSpan(MDSpan<const Type, Extents> values) const;

  template <class Type>
  void putArray(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void putArray(MDSpan<const Type, Extents> values) const;

  void putArray(Span<const String> values) const;

 public:

  template <class Type>
  void getSpan(Span2<Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void getSpan(MDSpan<Type, Extents> values) const;

  template <class Type>
  void getArray(Array2<Type>& values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void getArray(NumArray<Type, Extents>& values) const;

  void getArray(Array<String>& values) const;

 private:

  ISerializer* m_serializer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
reserveSpan(Span2<const Type> values) const
{
  Span<const Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->reserveSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
reserveSpan(MDSpan<const Type, Extents> values) const
{
  m_serializer->reserveSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
reserveArray(Span2<const Type> values) const
{
  m_serializer->reserveInt64(2);
  this->reserveSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
reserveArray(MDSpan<const Type, Extents> values) const
{
  constexpr Int32 rank = Extents::rank();
  m_serializer->reserveInt64(rank);
  m_serializer->reserveSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
putSpan(Span2<const Type> values) const
{
  Span<const Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->putSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
putSpan(MDSpan<const Type, Extents> values) const
{
  m_serializer->putSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
putArray(Span2<const Type> values) const
{
  m_serializer->putInt64(values.dim1Size());
  m_serializer->putInt64(values.dim2Size());
  this->putSpan(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
putArray(MDSpan<const Type, Extents> values) const
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

template <class Type>
void SerializerExtension::
getSpan(Span2<Type> values) const
{
  Span<Type> span_1d(values.data(), values.dim1Size() * values.dim2Size());
  m_serializer->getSpan(span_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type, class Extents>
requires(Extents::rank() <= 4) void SerializerExtension::
getSpan(MDSpan<Type, Extents> values) const
{
  Span<Type> values_1d = values.to1DSpan();
  m_serializer->getSpan(values_1d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void SerializerExtension::
getArray(Array2<Type>& values) const
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
getArray(NumArray<Type, Extents>& values) const
{
  using IndexType = Extents::ExtentIndexType;

  {
    constexpr Int32 rank = Extents::rank();
    std::array<IndexType, rank> size{};
    for (Int32 i = 0; i < rank; ++i) {
      size[i] = static_cast<IndexType>(m_serializer->getInt64());
    }
    MDIndex<rank, IndexType> mdi(size);
    values.resizeDestructive(mdi);
  }

  m_serializer->getSpan(values.to1DSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
