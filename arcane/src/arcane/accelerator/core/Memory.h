// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Memory.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Classes de gestion mémoire associées aux accélérateurs.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_MEMORY_H
#define ARCANE_ACCELERATOR_CORE_MEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/MemoryView.h"
#include "arccore/base/Span.h"

#include "arcane/accelerator/core/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conseils pour la gestion mémoire.
 */
enum class eMemoryAdvice
{
  //! Aucun conseil
  None = 0,
  //! Indique que la zone mémoire est principalement en lecture seule.
  MostlyRead,
  //! Privilégié le positionnement de la mémoire sur l'accélérateur
  PreferredLocationDevice,
  //! Privilégié le positionnement de la mémoire sur l'hôte.
  PreferredLocationHost,
  //! Indique que la zone mémoire est accédée par l'accélérateur.
  AccessedByDevice,
  //! Indique que la zone mémoire est accédée par l'hôte.
  AccessedByHost
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT std::ostream&
operator<<(std::ostream& o,eMemoryAdvice r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour la copie mémoire.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT MemoryCopyArgs
{
 private:

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
  : MemoryCopyArgs(_toSpan(destination, length),_toSpan(source, length))
  {}

  //! Copie \a source.size() octets depuis \a source vers \a destination
  MemoryCopyArgs(Span<std::byte> destination, Span<const std::byte> source)
  : m_source(source)
  , m_destination(destination)
  {
    // TODO: vérifier destination.size() > source.size();
  }

  //! Copie depuis \a source vers \a destination
  MemoryCopyArgs(MutableMemoryView destination, ConstMemoryView source)
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
  ConstMemoryView source() const { return m_source; }
  MutableMemoryView destination() const { return m_destination; }
  bool isAsync() const { return m_is_async; }

 private:

  ConstMemoryView m_source;
  MutableMemoryView m_destination;
  bool m_is_async = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour le préfetching mémoire.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT MemoryPrefetchArgs
{
 private:

  static Span<const std::byte> _toSpan(const void* ptr, Int64 length)
  {
    return { reinterpret_cast<const std::byte*>(ptr), length };
  }
  /*static Span<std::byte> _toSpan(void* ptr, Int64 length)
  {
    return { reinterpret_cast<std::byte*>(ptr), length };
    }*/

 public:

  //! Prefetch \a length octets depuis \a source
  MemoryPrefetchArgs(const void* source, Int64 length)
  : MemoryPrefetchArgs(_toSpan(source, length))
  {}

  //! Prefetch \a source
  explicit MemoryPrefetchArgs(ConstMemoryView source)
  : m_source(source)
  {}

  //! Prefetch \a source
  explicit MemoryPrefetchArgs(Span<const std::byte> source)
  : m_source(ConstMemoryView(source))
  {}

 public:

  MemoryPrefetchArgs& addAsync()
  {
    m_is_async = true;
    return (*this);
  }
  MemoryPrefetchArgs& addAsync(bool v)
  {
    m_is_async = v;
    return (*this);
  }
  MemoryPrefetchArgs& addDeviceId(DeviceId v)
  {
    m_device_id = v;
    return (*this);
  }
  ConstMemoryView source() const { return m_source; }
  bool isAsync() const { return m_is_async; }
  DeviceId deviceId() const { return m_device_id; }

 private:

  ConstMemoryView m_source;
  DeviceId m_device_id;
  bool m_is_async = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
