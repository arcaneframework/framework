// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.h                                                (C) 2000-2025 */
/*                                                                           */
/* Constant or mutable views on a memory region.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MEMORYVIEW_H
#define ARCCORE_BASE_MEMORYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup MemoryView
 * \brief Constant view on a contiguous memory region containing
 * fixed-size elements.
 *
 * The makeConstMemoryView() functions allow creating instances
 * of this class.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 */
class ARCCORE_BASE_EXPORT ConstMemoryView
{
  friend ARCCORE_BASE_EXPORT ConstMemoryView
  makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element);

 public:

  using SpanType = Span<const std::byte>;
  friend MutableMemoryView;

 public:

  ConstMemoryView() = default;
  explicit constexpr ConstMemoryView(Span<const std::byte> bytes)
  : m_bytes(bytes)
  , m_nb_element(bytes.size())
  , m_datatype_size(1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(Span<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(Span<const DataType> v)
  : ConstMemoryView(v, 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(ConstArrayView<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr ConstMemoryView(ArrayView<DataType> v)
  : ConstMemoryView(Span<const DataType>(v), 1)
  {}
  template <typename DataType> constexpr ConstMemoryView(ConstArrayView<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr ConstMemoryView(ArrayView<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr ConstMemoryView(Span<DataType> v, Int32 nb_component)
  : ConstMemoryView(Span<const DataType>(v), nb_component)
  {
  }
  template <typename DataType> constexpr ConstMemoryView(Span<const DataType> v, Int32 nb_component)
  : m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {
    auto x = asBytes(v);
    m_bytes = SpanType(x.data(), x.size() * nb_component);
  }

 public:

  template <typename DataType> constexpr ConstMemoryView&
  operator=(Span<DataType> v)
  {
    m_bytes = asBytes(v);
    m_nb_element = v.size();
    m_datatype_size = static_cast<Int32>(sizeof(DataType));
    return (*this);
  }

 private:

  constexpr ConstMemoryView(Span<const std::byte> bytes, Int32 datatype_size, Int64 nb_element)
  : m_bytes(bytes)
  , m_nb_element(nb_element)
  , m_datatype_size(datatype_size)
  {}

 public:

  //! View in byte form
  constexpr SpanType bytes() const { return m_bytes; }

  //! Pointer to the memory region
  constexpr const std::byte* data() const { return m_bytes.data(); }

  //! Number of elements
  constexpr Int64 nbElement() const { return m_nb_element; }

  //! Size of the associated data type (1 by default)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

  //! Sub-view starting from index \a begin_index and containing \a nb_element
  constexpr ConstMemoryView subView(Int64 begin_index, Int64 nb_element) const
  {
    Int64 byte_offset = begin_index * m_datatype_size;
    auto sub_bytes = m_bytes.subspan(byte_offset, nb_element * m_datatype_size);
    return { sub_bytes, m_datatype_size, nb_element };
  }

 public:

  //! View converted to a Span
  ARCCORE_DEPRECATED_REASON("Use bytes() instead")
  SpanType span() const { return m_bytes; }

  ARCCORE_DEPRECATED_REASON("Use bytes().size() instead")
  constexpr Int64 size() const { return m_bytes.size(); }

 private:

  SpanType m_bytes;
  Int64 m_nb_element = 0;
  Int32 m_datatype_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup MemoryView
 *
 * \brief Mutable view on a contiguous memory region containing
 * fixed-size elements.
 *
 * The makeMutableMemoryView() functions allow creating instances
 * of this class.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 */
class ARCCORE_BASE_EXPORT MutableMemoryView
{
  friend ARCCORE_BASE_EXPORT MutableMemoryView
  makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element);

 public:

  using SpanType = Span<std::byte>;

 public:

  MutableMemoryView() = default;
  explicit constexpr MutableMemoryView(SpanType bytes)
  : m_bytes(bytes)
  , m_nb_element(bytes.size())
  , m_datatype_size(1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(Span<DataType> v)
  : MutableMemoryView(v, 1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(ArrayView<DataType> v)
  : MutableMemoryView(Span<DataType>(v), 1)
  {}
  template <typename DataType> explicit constexpr MutableMemoryView(ArrayView<DataType> v, Int32 nb_component)
  : MutableMemoryView(Span<DataType>(v), nb_component)
  {}
  template <typename DataType> constexpr MutableMemoryView(Span<DataType> v, Int32 nb_component)
  : m_nb_element(v.size())
  , m_datatype_size(static_cast<Int32>(sizeof(DataType)) * nb_component)
  {
    auto x = asWritableBytes(v);
    m_bytes = SpanType(x.data(), x.size() * nb_component);
  }

 public:

  template <typename DataType> constexpr MutableMemoryView&
  operator=(Span<DataType> v)
  {
    m_bytes = asWritableBytes(v);
    m_nb_element = v.size();
    m_datatype_size = static_cast<Int32>(sizeof(DataType));
    return (*this);
  }

 private:

  constexpr MutableMemoryView(Span<std::byte> bytes, Int32 datatype_size, Int64 nb_element)
  : m_bytes(bytes)
  , m_nb_element(nb_element)
  , m_datatype_size(datatype_size)
  {}

 public:

  constexpr operator ConstMemoryView() const { return { m_bytes, m_datatype_size, m_nb_element }; }

 public:

  //! View in byte form
  constexpr SpanType bytes() const { return m_bytes; }

  //! Pointer to the memory region
  constexpr std::byte* data() const { return m_bytes.data(); }

  //! Number of elements
  constexpr Int64 nbElement() const { return m_nb_element; }

  //! Size of the associated data type (1 by default)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

  //! Sub-view starting from index \a begin_index
  constexpr MutableMemoryView subView(Int64 begin_index, Int64 nb_element) const
  {
    Int64 byte_offset = begin_index * m_datatype_size;
    auto sub_bytes = m_bytes.subspan(byte_offset, nb_element * m_datatype_size);
    return { sub_bytes, m_datatype_size, nb_element };
  }

 public:

  ARCCORE_DEPRECATED_REASON("Use bytes() instead")
  constexpr SpanType span() const { return m_bytes; }

  ARCCORE_DEPRECATED_REASON("Use bytes().size() instead")
  constexpr Int64 size() const { return m_bytes.size(); }

 private:

  SpanType m_bytes;
  Int64 m_nb_element = 0;
  Int32 m_datatype_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup MemoryView
 *
 * \brief List of constant views on contiguous memory regions.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 */
class ARCCORE_BASE_EXPORT ConstMultiMemoryView
{
 public:

  ConstMultiMemoryView(SmallSpan<const Span<const std::byte>> views, Int32 datatype_size)
  : m_views(views)
  , m_datatype_size(datatype_size)
  {}
  ConstMultiMemoryView(SmallSpan<const Span<std::byte>> views, Int32 datatype_size)
  : m_datatype_size(datatype_size)
  {
    auto* ptr = reinterpret_cast<const Span<const std::byte>*>(views.data());
    m_views = { ptr, views.size() };
  }

  //! Views in byte form on the memory region
  constexpr SmallSpan<const Span<const std::byte>> views() const { return m_views; }

  //! Size of the associated data type (1 by default)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

 private:

  SmallSpan<const Span<const std::byte>> m_views;
  Int32 m_datatype_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup MemoryView
 *
 * \brief List of mutable views on contiguous memory regions.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 */
class ARCCORE_BASE_EXPORT MutableMultiMemoryView
{
 public:

  MutableMultiMemoryView(SmallSpan<Span<std::byte>> views, Int32 datatype_size)
  : m_views(views)
  , m_datatype_size(datatype_size)
  {}

 public:

  //! Views in byte form on the memory region
  constexpr SmallSpan<Span<std::byte>> views() const { return m_views; }

  //! Size of the associated data type (1 by default)
  constexpr Int32 datatypeSize() const { return m_datatype_size; }

 private:

  SmallSpan<Span<std::byte>> m_views;
  Int32 m_datatype_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Creates a constant memory view from a \a Span
template <typename DataType> ConstMemoryView
makeMemoryView(Span<DataType> v)
{
  return ConstMemoryView(v);
}

//! Creates a constant memory view at address \a v
template <typename DataType> ConstMemoryView
makeMemoryView(const DataType* v)
{
  return ConstMemoryView(Span<const DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Creates a mutable memory view from a \a Span
template <typename DataType> MutableMemoryView
makeMutableMemoryView(Span<DataType> v)
{
  return MutableMemoryView(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Creates a mutable memory view at address \a v
template <typename DataType> MutableMemoryView
makeMutableMemoryView(DataType* v)
{
  return MutableMemoryView(Span<DataType>(v, 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a mutable memory view.
 *
 * \param ptr address of the memory region.
 * \param datatype_size size (in bytes) of the data type.
 * \param nb_element number of elements in the view.
 *
 * The memory region will have a size of datatype_size * nb_element bytes.
 */
extern "C++" ARCCORE_BASE_EXPORT MutableMemoryView
makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a read-only memory view.
 *
 * \param ptr address of the memory region.
 * \param datatype_size size (in bytes) of the data type.
 * \param nb_element number of elements in the view.
 *
 * The memory region will have a size of datatype_size * nb_element bytes.
 */
extern "C++" ARCCORE_BASE_EXPORT ConstMemoryView
makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
