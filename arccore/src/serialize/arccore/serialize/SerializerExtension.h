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

/*!
 * \brief Class that extends the functionalities of the ISerializer.
 *
 * This class allows you to serialize multidimensionnal arrays or String arrays.
 * It uses methods of the classical ISerializer. You can destroy this class
 * between steps of serialization, it doesn't store any datas.
 */
class ARCCORE_SERIALIZE_EXPORT SerializerExtension
{
 public:

  /*!
   * \brief Constructor
   * \param serializer The classical serializer.
   */
  SerializerExtension(ISerializer* serializer)
  : m_serializer(serializer)
  {}

 public:

  //! Reserve for a view of \a values elements
  template <class Type>
  void reserveSpan(Span2<const Type> values) const;

  //! Reserve for a view of \a values elements
  template <class Type, class Extents>
  void reserveSpan(MDSpan<const Type, Extents> values) const
  requires(Extents::rank() <= 4);

  //! Reserve to save the number of elements and the \a values elements
  template <class Type>
  void reserveArray(Span2<const Type> values) const;

  //! Reserve to save the number of elements and the \a values elements
  template <class Type, class Extents>
  void reserveArray(MDSpan<const Type, Extents> values) const
  requires(Extents::rank() <= 4);

  //! Reserve to save the number of elements and the \a values elements
  void reserveArray(Span<const String> values) const;

 public:

  //! Add the array \a values
  template <class Type>
  void putSpan(Span2<const Type> values) const;

  //! Add the array \a values
  template <class Type, class Extents>
  void putSpan(MDSpan<const Type, Extents> values) const
  requires(Extents::rank() <= 4);

  //! Save the number of elements and the \a values elements
  template <class Type>
  void putArray(Span2<const Type> values) const;

  //! Save the number of elements and the \a values elements
  template <class Type, class Extents>
  void putArray(MDSpan<const Type, Extents> values) const
  requires(Extents::rank() <= 4);

  //! Save the number of elements and the \a values elements
  void putArray(Span<const String> values) const;

 public:

  //! Retrieve the array \a values
  template <class Type>
  void getSpan(Span2<Type> values) const;

  //! Retrieve the array \a values
  template <class Type, class Extents>
  void getSpan(MDSpan<Type, Extents> values) const
  requires(Extents::rank() <= 4);

  //! Resize and fill \a values
  template <class Type>
  void getArray(Array2<Type>& values) const;

  //! Resize and fill \a values
  template <class Type, class Extents>
  void getArray(NumArray<Type, Extents>& values) const
  requires(Extents::rank() <= 4);

  //! Resize and fill \a values
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
void SerializerExtension::
reserveSpan(MDSpan<const Type, Extents> values) const
requires(Extents::rank() <= 4)
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
void SerializerExtension::
reserveArray(MDSpan<const Type, Extents> values) const
requires(Extents::rank() <= 4)
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
void SerializerExtension::
putSpan(MDSpan<const Type, Extents> values) const
requires(Extents::rank() <= 4)
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
void SerializerExtension::
putArray(MDSpan<const Type, Extents> values) const
requires(Extents::rank() <= 4)
{
  constexpr Int32 rank = Extents::rank();
  // TODO : Add method in ArrayExtents to get an std::array to remove
  //        all 'requires()' of this class and simplify this method.
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
void SerializerExtension::
getSpan(MDSpan<Type, Extents> values) const
requires(Extents::rank() <= 4)
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
void SerializerExtension::
getArray(NumArray<Type, Extents>& values) const
requires(Extents::rank() <= 4)
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
