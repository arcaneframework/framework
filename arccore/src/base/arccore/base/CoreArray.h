// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CoreArray.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Simple array for Arccore.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_COREARRAY_H
#define ARCCORE_BASE_COREARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal array for Arccore.
 *
 * This class is private and should only be used by Arccore classes.
 */
template <class DataType>
class CoreArray
{
 private:

  typedef std::vector<DataType> ContainerType;

 public:

  //! Type of the array elements
  typedef DataType value_type;
  //! Type of the iterator over an array element
  typedef typename ContainerType::iterator iterator;
  //! Type of the constant iterator over an array element
  typedef typename ContainerType::const_iterator const_iterator;
  //! Type pointer of an array element
  typedef typename ContainerType::pointer pointer;
  //! Type constant pointer of an array element
  typedef const value_type* const_pointer;
  //! Type reference of an array element
  typedef value_type& reference;
  //! Type constant reference of an array element
  typedef const value_type& const_reference;
  //! Type indexing the array
  typedef Int64 size_type;
  //! Type of a distance between array element iterators
  typedef ptrdiff_t difference_type;

 public:

  //! Constructs an empty array.
  CoreArray() {}
  //! Constructs an empty array.
  CoreArray(ConstArrayView<DataType> v)
  : m_p(v.begin(), v.end())
  {}
  CoreArray(Span<const DataType> v)
  : m_p(v.begin(), v.end())
  {}

 public:

  //! Conversion to a Span<const DataType>
  operator Span<const DataType>() const
  {
    return CoreArray::_constSpan(m_p);
  }
  //! Conversion to a Span<DataType>
  operator Span<DataType>()
  {
    return CoreArray::_span(m_p);
  }

 public:

  //! i-th element of the array.
  inline DataType& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i, m_p.size());
    return m_p[i];
  }

  //! i-th element of the array.
  inline const DataType& operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_p.size());
    return m_p[i];
  }

  //! Returns the size of the array
  inline Int64 size() const { return static_cast<Int64>(m_p.size()); }

  //! Returns an iterator to the first element of the array
  inline iterator begin() { return m_p.begin(); }
  //! Returns an iterator to the first element after the end of the array
  inline iterator end() { return m_p.end(); }
  //! Returns a constant iterator to the first element of the array
  inline const_iterator begin() const { return m_p.begin(); }
  //! Returns a constant iterator to the first element after the end of the array
  inline const_iterator end() const { return m_p.end(); }

  //! Constant view
  ConstArrayView<DataType> constView() const
  {
    return CoreArray::_constView(m_p);
  }

  //! Modifiable view
  ArrayView<DataType> view()
  {
    return CoreArray::_view(m_p);
  }

  //! Constant view
  Span<const DataType> constSpan() const
  {
    return CoreArray::_constSpan(m_p);
  }

  //! Modifiable view
  Span<DataType> span()
  {
    return CoreArray::_span(m_p);
  }

  //! Returns true if the array is empty
  bool empty() const
  {
    return m_p.empty();
  }

  void resize(Int64 new_size)
  {
    m_p.resize(new_size);
  }
  void reserve(Int64 new_size)
  {
    m_p.reserve(new_size);
  }
  void clear()
  {
    m_p.clear();
  }
  void add(const DataType& v)
  {
    CoreArray::_add(m_p, v);
  }
  DataType& back()
  {
    return m_p.back();
  }
  const DataType& back() const
  {
    return m_p.back();
  }
  const DataType* data() const
  {
    return _data(m_p);
  }
  DataType* data()
  {
    return _data(m_p);
  }
  bool contains(const_reference v) const
  {
    for (const auto& x : m_p)
      if (x == v)
        return true;
    return false;
  }

  /*!
   * \brief Removes the element with value \a v from the list.
   *
   * Only the first instance of the element with value \a v
   * is removed. If the value is not found, no operation
   * is performed.
   */
  void removeValue(const_reference v)
  {
    auto e = m_p.end();
    for (auto b = m_p.begin(); b != e; ++b)
      if (*b == v) {
        m_p.erase(b);
        return;
      }
  }

 private:

  static ConstArrayView<DataType> _constView(const std::vector<DataType>& c)
  {
    Int32 s = arccoreCheckArraySize(c.size());
    return ConstArrayView<DataType>(s, c.data());
  }
  static ArrayView<DataType> _view(std::vector<DataType>& c)
  {
    Int32 s = arccoreCheckArraySize(c.size());
    return ArrayView<DataType>(s, c.data());
  }
  static Span<const DataType> _constSpan(const std::vector<DataType>& c)
  {
    Int64 s = static_cast<Int64>(c.size());
    return Span<const DataType>(c.data(), s);
  }
  static Span<DataType> _span(std::vector<DataType>& c)
  {
    Int64 s = static_cast<Int64>(c.size());
    return Span<DataType>(c.data(), s);
  }
  static void _add(std::vector<DataType>& c, const DataType& v)
  {
    c.push_back(v);
  }
  static DataType* _data(std::vector<DataType>& c)
  {
    return c.data();
  }
  static const DataType* _data(const std::vector<DataType>& c)
  {
    return c.data();
  }

 private:

  ContainerType m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
