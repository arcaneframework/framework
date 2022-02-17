// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Memory.h                                                    (C) 2000-2022 */
/*                                                                           */
/* Classes ayant trait à la gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_MEMORY_H
#define ARCANE_ACCELERATOR_CORE_MEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/utils/UtilsTypes.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour la copie mémoire.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT MemoryCopyArgs
{
 public:

  static Span<const std::byte> _toSpan(const void* ptr, Int64 length)
  {
    return { reinterpret_cast<const std::byte*>(ptr), length };
  }
  static Span<std::byte> _toSpan(void* ptr, Int64 length)
  {
    return { reinterpret_cast<std::byte*>(ptr), length };
  }

 public:

  //! Copie \a length octets depuis \a source vers \a destination
  MemoryCopyArgs(void* destination, const void* source, Int64 length)
  : m_source(_toSpan(source, length))
  , m_destination(_toSpan(destination, length))
  {}

  //! Copie depuis \a source vers \a destination
  MemoryCopyArgs(Span<std::byte> destination, Span<const std::byte> source)
  : m_source(source)
  , m_destination(destination)
  {}

 public:

  MemoryCopyArgs& addAsync()
  {
    m_is_async = true;
    return (*this);
  }
  MemoryCopyArgs& addAsync(bool v)
  {
    m_is_async = v;
    return (*this);
  }
  Span<const std::byte> source() const { return m_source; }
  Span<std::byte> destination() const { return m_destination; }
  bool isAsync() const { return m_is_async; }

 private:

  Span<const std::byte> m_source;
  Span<std::byte> m_destination;
  bool m_is_async = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
