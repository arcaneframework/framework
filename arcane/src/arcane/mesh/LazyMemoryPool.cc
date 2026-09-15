// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LazyMemoryPool.cc                                           (C) 2000-2026 */
/*                                                                           */
/* Zero buffers shared by the connectivities that are still lazy.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/internal/LazyMemoryPool.h"

#include "arcane/utils/MemoryUtils.h"

#include <utility>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LazyMemoryPool::Buffer::
Buffer()
: m_zeros(MemoryUtils::getDefaultDataAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> LazyMemoryPool::
grow(Buffer*& buffer, Int32 n)
{
  std::lock_guard<std::mutex> lk(m_mutex);
  if (buffer && n <= buffer->m_zeros.size())
    return buffer->m_zeros.constView();
  Buffer* new_buffer = _smallestBuffer(n, false);
  if (!new_buffer) {
    Int32 size = 1024;
    while (size < n)
      size *= 2;
    new_buffer = _addBuffer(size, false);
  }
  return _moveTo(buffer, new_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> LazyMemoryPool::
trim(Buffer*& buffer, Int32 n)
{
  std::lock_guard<std::mutex> lk(m_mutex);
  // Another connectivity, possibly of another mesh, may already have allocated
  // a buffer that fits: sharing it keeps one buffer for all of them.
  Buffer* new_buffer = _smallestBuffer(n, true);
  if (!new_buffer)
    new_buffer = _addBuffer(n, true);
  return _moveTo(buffer, new_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LazyMemoryPool::
release(Buffer*& buffer)
{
  std::lock_guard<std::mutex> lk(m_mutex);
  _release(buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LazyMemoryPool::Buffer* LazyMemoryPool::
_smallestBuffer(Int32 n, bool only_exact)
{
  Buffer* best = nullptr;
  for (const auto& b : m_buffers) {
    if (only_exact && !b->m_is_exact)
      continue;
    if (b->m_zeros.size() < n)
      continue;
    if (!best || b->m_zeros.size() < best->m_zeros.size())
      best = b.get();
  }
  return best;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LazyMemoryPool::Buffer* LazyMemoryPool::
_addBuffer(Int32 size, bool is_exact)
{
  auto b = std::make_unique<Buffer>();
  b->m_zeros.resize(size);
  b->m_zeros.fill(0);
  b->m_is_exact = is_exact;
  Buffer* p = b.get();
  m_buffers.push_back(std::move(b));
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Moves the reader of \a buffer to \a new_buffer.
 *
 * The old buffer may be freed here, while the caller's views still point into
 * it: the caller republishes them right after, from the only thread that reads
 * them.
 */
ConstArrayView<Int32> LazyMemoryPool::
_moveTo(Buffer*& buffer, Buffer* new_buffer)
{
  if (buffer != new_buffer) {
    ++new_buffer->m_nb_reader;
    _release(buffer);
    buffer = new_buffer;
  }
  return new_buffer->m_zeros.constView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LazyMemoryPool::
_release(Buffer*& buffer)
{
  if (!buffer)
    return;
  --buffer->m_nb_reader;
  if (buffer->m_nb_reader == 0) {
    for (auto& b : m_buffers) {
      if (b.get() == buffer) {
        std::swap(b, m_buffers.back());
        m_buffers.pop_back();
        break;
      }
    }
  }
  buffer = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
