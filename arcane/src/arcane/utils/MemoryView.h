// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.h                                                (C) 2000-2022 */
/*                                                                           */
/* Vue constantes ou modifiables sur une zone mémoire.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYVIEW_H
#define ARCANE_UTILS_MEMORYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue constante sur une zone mémoire.
 */
class ARCANE_UTILS_EXPORT MemoryView
{
 public:

  using SpanType = Span<const std::byte>;

 public:

  MemoryView() = default;
  explicit constexpr MemoryView(Span<const std::byte> bytes)
  : m_bytes(bytes)
  {}

 public:

  //! Vue convertie en un Span
  SpanType span() const { return m_bytes; }
  constexpr Int64 size() const { return m_bytes.size(); }

 public:

  SpanType m_bytes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une collection fortement typée.
 * \ingroup MemoryView
 */
class ARCANE_UTILS_EXPORT MutableMemoryView
{
 public:

  using SpanType = Span<std::byte>;

 public:

  MutableMemoryView() = default;
  explicit constexpr MutableMemoryView(SpanType bytes)
  : m_bytes(bytes)
  {}

 public:

   operator MemoryView() const { return MemoryView(m_bytes); }

 public:

  //! Vue convertie en un Span
  constexpr SpanType span() const { return m_bytes; }
  constexpr Int64 size() const { return m_bytes.size(); }

 public:

  SpanType m_bytes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire constante à partir d'un \a Span
template<typename DataType> MemoryView
makeMemoryView(Span<DataType> v)
{
  auto bytes = asBytes(Span<const DataType>(v));
  return MemoryView(bytes);
}

//! Créé une vue mémoire constante sur l'adresse \a v
template<typename DataType> MemoryView
makeMemoryView(const DataType* v)
{
  const Int64 s = (Int64)(sizeof(DataType));
  const std::byte* ptr = reinterpret_cast<const std::byte*>(v);
  return MemoryView(Span<const std::byte>(ptr,s));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une vue mémoire modifiable à partir d'un \a Span
template<typename DataType> MutableMemoryView
makeMutableMemoryView(Span<DataType> v)
{
  auto bytes = asWritableBytes(v);
  return MutableMemoryView(bytes);
}

//! Créé une vue mémoire modifiable sur l'adresse \a v
template<typename DataType> MutableMemoryView
makeMutableMemoryView(DataType* v)
{
  const Int64 s = (Int64)(sizeof(DataType));
  std::byte* ptr = reinterpret_cast<std::byte*>(v);
  return MutableMemoryView(Span<std::byte>(ptr,s));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
