// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Memory.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Memory management classes associated with accelerators.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_MEMORY_H
#define ARCCORE_COMMON_ACCELERATOR_MEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/MemoryView.h"
#include "arccore/base/Span.h"

#include "arccore/common/accelerator/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory management advice.
 */
enum class eMemoryAdvice
{
  //! No advice
  None = 0,
  //! Indicates that the memory region is primarily read-only.
  MostlyRead,
  //! Prefers memory placement on the accelerator
  PreferredLocationDevice,
  //! Prefers memory placement on the host.
  PreferredLocationHost,
  //! Indicates that the memory region is accessed by the device.
  AccessedByDevice,
  //! Indicates that the memory region is accessed by the host.
  AccessedByHost
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryAdvice r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory copy arguments.
 */
class ARCCORE_COMMON_EXPORT MemoryCopyArgs
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

  //! Copies \a length bytes from \a source to \a destination
  MemoryCopyArgs(void* destination, const void* source, Int64 length)
  : MemoryCopyArgs(_toSpan(destination, length), _toSpan(source, length))
  {}

  //! Copies \a source.size() bytes from \a source to \a destination
  MemoryCopyArgs(Span<std::byte> destination, Span<const std::byte> source)
  : m_source(source)
  , m_destination(destination)
  {
    // TODO: vérifier destination.size() > source.size();
  }

  //! Copies from \a source to \a destination
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
 * \brief Memory prefetching arguments.
 */
class ARCCORE_COMMON_EXPORT MemoryPrefetchArgs
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

  //! Prefetches \a length bytes from \a source
  MemoryPrefetchArgs(const void* source, Int64 length)
  : MemoryPrefetchArgs(_toSpan(source, length))
  {}

  //! Prefetches \a source
  explicit MemoryPrefetchArgs(ConstMemoryView source)
  : m_source(source)
  {}

  //! Prefetches \a source
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
