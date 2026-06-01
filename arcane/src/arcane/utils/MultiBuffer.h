// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiBuffer.h                                               (C) 2000-2011 */
/*                                                                           */
/* Template class of an array with a buffer.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIBUFFER_H
#define ARCANE_UTILS_MULTIBUFFER_H
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
 * \internal
 * \brief Buffer for multiple allocation

 This class manages a pre-allocated list of elements in order to limit
 multiple calls to allocations (new() or malloc()).

 Pre-allocations are done in blocks of \a m_buffer_size elements

 To be used by this class, a type must possess a default constructor
 and a copy operator. This class guarantees that the
 returned pointers remain valid as long as this instance exists.

 The constructors and copy operators do not duplicate memory
 but simply retain the buffer size.
*/
template <class T>
class MultiBufferT
{
 public:

  typedef UniqueArray<T> BufferType;

 public:

  MultiBufferT()
  : m_buffer_size(1000)
  , m_current_buffer_size(0)
  , m_nb_in_buffer(0)
  , m_current_buffer(0)
  {
  }
  MultiBufferT(Integer buf_size)
  : m_buffer_size(buf_size)
  , m_current_buffer_size(0)
  , m_nb_in_buffer(0)
  , m_current_buffer(0)
  {
  }

  //! Copy constructor
  MultiBufferT(const MultiBufferT<T>& ref)
  : m_buffer_size(ref.m_buffer_size)
  , m_current_buffer_size(0)
  , m_nb_in_buffer(0)
  , m_current_buffer(0)
  {
  }
  ~MultiBufferT()
  {
    _freeAllocatedBuffers();
  }

 public:

  //! Copy assignment operator (forbidden)
  void operator=(const MultiBufferT<T>& ref)
  {
    if (&ref == this)
      return;
    clear();
    m_buffer_size = ref.m_buffer_size;
  }

 public:

  //! Allocates a new element
  T* allocOne()
  {
    if (!m_current_buffer)
      _allocateCurrentBuffer();
    T* v = &(*m_current_buffer)[m_nb_in_buffer];
    ++m_nb_in_buffer;
    if (m_nb_in_buffer == m_current_buffer_size)
      m_current_buffer = 0; // Indicates that the current buffer is full
    return v;
  }

  //! Allocates \a n elements
  ArrayView<T> allocMany(Integer n)
  {
    // If the desired number of elements is greater than the size
    // of the buffer, specifically allocate a buffer of the correct size
    if (n > m_current_buffer_size) {
      BufferType* bt = new BufferType(n);
      m_allocated_buffers.add(bt);
      return *bt;
    }
    if (!m_current_buffer)
      _allocateCurrentBuffer();
    // If the current buffer is not large enough to contain the
    // \a n desired elements, allocate another one
    if ((m_nb_in_buffer + n) >= m_current_buffer_size)
      _allocateCurrentBuffer();
    T* v = &(*m_current_buffer)[m_nb_in_buffer];
    m_nb_in_buffer += n;
    if (m_nb_in_buffer == m_current_buffer_size)
      m_current_buffer = 0; // Indicates that the current buffer is full
    return ArrayView<T>(n, v);
  }
  void clear()
  {
    m_nb_in_buffer = 0;
    m_current_buffer = 0;
    m_current_buffer_size = 0;
    _freeAllocatedBuffers();
  }
  Integer nbAllocatedBuffer() { return m_allocated_buffers.size(); }
  Integer bufferSize() const { return m_current_buffer_size; }

 protected:

  void _freeAllocatedBuffers()
  {
    for (Integer i = 0, s = m_allocated_buffers.size(); i < s; ++i)
      delete m_allocated_buffers[i];
    m_allocated_buffers.clear();
  }

 private:

  Integer m_buffer_size; //!< Number of elements in a buffer
  Integer m_current_buffer_size; //!< Maximum number of elements in the current buffer
  Integer m_nb_in_buffer; //!< Number of elements in the current buffer.
  BufferType* m_current_buffer; //!< Current buffer
  UniqueArray<BufferType*> m_allocated_buffers; //!< List of all buffers
 private:

  void _allocateCurrentBuffer()
  {
    m_current_buffer = new BufferType(m_buffer_size);
    m_current_buffer_size = m_buffer_size;
    m_allocated_buffers.add(m_current_buffer);
    m_nb_in_buffer = 0;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
