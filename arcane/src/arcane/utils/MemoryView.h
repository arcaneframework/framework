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

#include "arcane/utils/ArcaneGlobal.h"

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

  //! Vue convertie en un Span
  constexpr SpanType span() const { return m_bytes; }

 public:

  SpanType m_bytes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
