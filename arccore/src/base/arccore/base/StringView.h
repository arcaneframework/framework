// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringView.h                                                (C) 2000-2025 */
/*                                                                           */
/* View of a UTF-8 character string.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGVIEW_H
#define ARCCORE_BASE_STRINGVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/Span.h"

#include <string_view>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of a UTF-8 character string.
 *
 * This class is similar to std::string_view in C++17 in that it only holds
 * a pointer to memory managed by another class.
 * Therefore, instances of this class should not be kept. The
 * main difference lies in the encoding, which must be UTF-8 with this class.
 *
 * \note Like the std::string_view class, the \a bytes() array does not
 * necessarily contain a null terminator. This means, among other things,
 * that this class should not be used to pass parameters to C functions.
 */
class ARCCORE_BASE_EXPORT StringView
{
 public:

  //! Creates a view of an empty string
  StringView() = default;
  //! Creates a view from \a str encoded in UTF-8. \a str may be null.
  StringView(const char* str) ARCCORE_NOEXCEPT
  : StringView(str ? std::string_view(str) : std::string_view()){}
  //! Creates a string from \a str in UTF-8 encoding
  StringView(std::string_view str) ARCCORE_NOEXCEPT
  : m_v(reinterpret_cast<const Byte*>(str.data()),str.size()){}
  //! Creates a string from \a str in UTF-8 encoding
  constexpr StringView(Span<const Byte> str) ARCCORE_NOEXCEPT
  : m_v(str){}
  //! Copy constructor
  constexpr StringView(const StringView& str) = default;
  //! Copies the view \a str into this instance.
  constexpr StringView& operator=(const StringView& str) = default;
  //! Creates a view from \a str encoded in UTF-8
  StringView& operator=(const char* str) ARCCORE_NOEXCEPT
  {
    operator=(str ? std::string_view(str) : std::string_view());
    return (*this);
  }
  //! Creates a view from \a str encoded in UTF-8
  StringView& operator=(std::string_view str) ARCCORE_NOEXCEPT
  {
    m_v = Span<const Byte>(reinterpret_cast<const Byte*>(str.data()),str.size());
    return (*this);
  }
  //! Creates a view from \a str encoded in UTF-8
  constexpr StringView& operator=(Span<const Byte> str) ARCCORE_NOEXCEPT
  {
    m_v = str;
    return (*this);
  }

  ~StringView() = default; //!< Frees resources.

 public:

  /*!
   * \brief Returns the conversion of the instance in UTF-8 encoding.
   *
   * \warning The returned instance does not contain a null terminator.
   *
   * \warning The instance remains the owner of the returned value, and this value
   * is invalidated by any modification of this instance.
   */
  constexpr Span<const Byte> bytes() const ARCCORE_NOEXCEPT { return m_v; }

  //! Length in bytes of the character string.
  constexpr Int64 length() const ARCCORE_NOEXCEPT { return m_v.size(); }

  //! Length in bytes of the character string.
  constexpr Int64 size() const ARCCORE_NOEXCEPT { return m_v.size(); }

  //! True if the string is null or empty.
  constexpr bool empty() const ARCCORE_NOEXCEPT { return size()==0; }

 public:

  /*!
   * \brief Returns an STL view of the current view.
   */
  std::string_view toStdStringView() const ARCCORE_NOEXCEPT
  {
    return std::string_view(reinterpret_cast<const char*>(m_v.data()),m_v.size());
  }

  //! StringView output operator
  friend ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o,const StringView&);

  /*!
   * \brief Compares two views.
   * \retval true if they are equal,
   * \retval false otherwise.
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const StringView& a,const StringView& b);

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if they are different,
   * \retval false if they are equal.
   * \relate String
   */
  friend inline bool operator!=(const StringView& a,const StringView& b)
  {
    return !operator==(a,b);
  }

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if they are equal,
   * \retval false otherwise.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const char* a,const StringView& b);

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if they are different,
   * \retval false if they are equal.
   * \relate String
   */
  friend bool operator!=(const char* a,const StringView& b){ return !operator==(a,b); }

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if they are equal,
   * \retval false otherwise.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator==(const StringView& a,const char* b);

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if they are different,
   * \retval false if they are equal.
   * \relate String
   */
  friend inline bool operator!=(const StringView& a,const char* b)
  {
    return !operator==(a,b);
  }

  /*!
   * \brief Compares two Unicode strings.
   * \retval true if a<b
   * \retval false otherwise.
   * \relate String
   */
  friend ARCCORE_BASE_EXPORT bool operator<(const StringView& a,const StringView& b);

 public:

  //! Writes the string in UTF-8 format to the stream \a o
  void writeBytes(std::ostream& o) const;

  //! Substring starting at position \a pos
  StringView subView(Int64 pos) const;

  //! Substring starting at position \a pos and of length \a len
  StringView subView(Int64 pos,Int64 len) const;

 private:

  Span<const Byte> m_v;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
