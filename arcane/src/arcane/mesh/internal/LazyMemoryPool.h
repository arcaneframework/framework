// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LazyMemoryPool.h                                            (C) 2000-2026 */
/*                                                                           */
/* Zero buffers shared by the connectivities that are still lazy.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INTERNAL_LAZYMEMORYPOOL_H
#define ARCANE_MESH_INTERNAL_LAZYMEMORYPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include <memory>
#include <mutex>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Zeros backing every lazy connectivity view of the process.
 *
 * One pool, and not one buffer per connectivity: they would all hold the same
 * thing. A buffer is never resized once published; growing means moving the
 * connectivity to another buffer, because Arcane can build one mesh per
 * subdomain thread and reallocating under another thread's published view is a
 * use after free. Each buffer counts the connectivities reading it and is
 * freed when the last one leaves it.
 *
 * While a mesh is built its item count is unknown, so grow() allocates powers
 * of two: an exact size would mean one new buffer per item added. Once the mesh
 * is updated, trim() moves each connectivity to one buffer of exactly the size
 * needed, and the growth buffers are freed as their last reader leaves.
 *
 * The buffers use the default data allocator, so that a lazy view can be read
 * from an accelerator like any other connectivity view.
 */
class LazyMemoryPool
{
 public:

  //! One zero buffer and the number of connectivities reading it
  struct Buffer
  {
    Buffer();
    UniqueArray<Int32> m_zeros;
    Int32 m_nb_reader = 0;
    //! Allocated by trim(), at the exact size asked
    bool m_is_exact = false;
  };

 public:

  //! Moves \a buffer to a zero buffer of at least \a n elements if it is smaller
  ConstArrayView<Int32> grow(Buffer*& buffer, Int32 n);

  //! Moves \a buffer to the smallest exact buffer of at least \a n elements
  ConstArrayView<Int32> trim(Buffer*& buffer, Int32 n);

  //! Stops reading \a buffer, and frees it if nothing else reads it
  void release(Buffer*& buffer);

 private:

  std::vector<std::unique_ptr<Buffer>> m_buffers;
  std::mutex m_mutex;

 private:

  Buffer* _smallestBuffer(Int32 n, bool only_exact);
  Buffer* _addBuffer(Int32 size, bool is_exact);
  ConstArrayView<Int32> _moveTo(Buffer*& buffer, Buffer* new_buffer);
  void _release(Buffer*& buffer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
