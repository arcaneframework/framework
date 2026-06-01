// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryBuffer.h                                              (C) 2000-2023 */
/*                                                                           */
/* Memory buffer.                                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_MEMORYBUFFER_H
#define ARCANE_IMPL_MEMORYBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of a memory buffer.
 *
 */
class ARCANE_UTILS_EXPORT MemoryBuffer
{
 private:

  explicit MemoryBuffer(IMemoryAllocator* allocator)
  : m_buffer(allocator)
  {}

 public:

  /*!
  * \brief Creates an instance of \a MemoryBuffer.
  *
  * The \a allocator must remain valid throughout
  * the lifetime of the created instance.
  */
  static Ref<MemoryBuffer> create(IMemoryAllocator* allocator)
  {
    auto* memory = new MemoryBuffer(allocator);
    Ref<MemoryBuffer> ref_memory = makeRef<MemoryBuffer>(memory);
    return ref_memory;
  }

 public:

  /*!
   * \brief Resizes the memory area.
   *
   * No initialization is performed. If the size decreases
   * resize() has no effect.
   */
  void resize(Int64 new_size) { m_buffer.resize(new_size); }
  Span<const std::byte> bytes() const { return m_buffer; }
  Span<std::byte> bytes() { return m_buffer; }
  IMemoryAllocator* allocator() const { return m_buffer.allocator(); }

 private:

  //! Buffer containing the data.
  UniqueArray<std::byte> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
