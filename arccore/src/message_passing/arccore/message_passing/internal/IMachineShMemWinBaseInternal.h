// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMachineShMemWinBaseInternal.h                              (C) 2000-2026 */
/*                                                                           */
/* Class interface allowing the creation of memory windows for a computing   */
/* node.                                                                     */
/* The segments of these windows are not contiguous in memory and can be     */
/* resized.                                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_IMACHINESHMEMWINBASEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the creation of memory windows for a computing node.
 *
 * The segments of these windows will not be contiguous in memory and can be
 * resized (one window per process and one segment per window).
 *
 * Since the add() method may want to resize a segment, and this resizing is
 * a collective operation, calling add() is therefore a collective operation
 *
 * To have non-concurrent add() calls, this operation is only possible on our
 * segment.
 * To add elements to the segment of another subdomain, the
 * addToAnotherSegment() methods are available.
 *
 * All sizes used are in bytes. \a sizeof_type is used only by MPI (if used)
 * and for verification purposes.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMachineShMemWinBaseInternal
{
 public:

  virtual ~IMachineShMemWinBaseInternal() = default;

 public:

  /*!
   * \brief Method to get the size of an element in the window.
   *
   * Non-collective call.
   *
   * \return The size of an element.
   */
  virtual Int32 sizeofOneElem() const = 0;

  /*!
   * \brief Method to get the ranks that possess a segment in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;

  /*!
   * \brief Method to wait until all processes/threads of the node call
   * this method to continue execution.
   */
  virtual void barrier() const = 0;

  /*!
   * \brief Method to get a view of our segment.
   *
   * Non-collective call.
   *
   * \return A view.
   */
  virtual Span<std::byte> segmentView() = 0;

  /*!
   * \brief Method to get a view of the segment of another subdomain
   * of the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  virtual Span<std::byte> segmentView(Int32 rank) = 0;

  /*!
   * \brief Method to get a view of our segment
   *
   * Non-collective call.
   *
   * \return A view.
   */
  virtual Span<const std::byte> segmentConstView() const = 0;

  /*!
   * \brief Method to get a view of the segment of another subdomain of the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  virtual Span<const std::byte> segmentConstView(Int32 rank) const = 0;

  /*!
   * \brief Method to add elements into our segment.
   *
   * Collective call.
   *
   * \note Do not mix calls to this method with calls to addToAnotherSegment().
   *
   * If the segment is too small, it will be resized.
   *
   * Subdomains that do not wish to add elements can call the \a add() method
   * without parameters or this method with an empty view.
   *
   * \param elem The elements to add.
   */
  virtual void add(Span<const std::byte> elem) = 0;

  /*!
   * See \a add(Span<const std::byte> elem).
   */
  virtual void add() = 0;

  /*!
   * \brief Method to add elements into the segment of another subdomain.
   *
   * Collective call.
   *
   * \note Do not mix calls to this method with calls to add().
   *
   * Two subdomains must not add elements to the same subdomain segment.
   *
   * If the targeted segment is too small, it will be resized.
   *
   * Subdomains that do not wish to add elements can call the
   * \a addToAnotherSegment() method without parameters.
   *
   * \param rank The subdomain into which to add elements.
   * \param elem The elements to add.
   */
  virtual void addToAnotherSegment(Int32 rank, Span<const std::byte> elem) = 0;

  /*!
   * See \a addToAnotherSegment(Int32 rank, Span<const std::byte> elem).
   */
  virtual void addToAnotherSegment() = 0;

  /*!
   * \brief Method to reserve memory space in our segment.
   *
   * Collective call.
   *
   * This method does nothing if \a new_capacity is less than the memory space
   * already allocated for the segment.
   * For processes that do not wish to increase the available memory space for
   * their segment, it is possible to set the \a new_capacity parameter to 0 or
   * use the \a reserve() method (without arguments).
   *
   * MPI will reserve a space with a size greater than or equal to \a new_capacity.
   *
   * This method does not resize the segment; you must always use the add() method
   * to add elements.
   *
   * To resize the segment, the \a resize(Int64 new_size) method is available.
   *
   * \param new_capacity The requested new capacity.
   */
  virtual void reserve(Int64 new_capacity) = 0;

  /*!
   * See \a reserve(Int64 new_capacity)
   */
  virtual void reserve() = 0;

  /*!
   * \brief Method to resize our segment.
   *
   * Collective call.
   *
   * If the provided size is less than the current size of the segment,
   * elements located after the provided size will be deleted.
   *
   * For processes that do not wish to resize their segment, it is possible
   * to set the \a new_size argument to -1 or call the \a resize() method
   * (without arguments).
   *
   * \param new_size The new size.
   */
  virtual void resize(Int64 new_size) = 0;

  /*!
   * See \a resize(Int64 new_size)
   */
  virtual void resize() = 0;

  /*!
   * \brief Method to reduce the reserved memory space for the segments
   * to the minimum necessary.
   *
   * Collective call.
   */
  virtual void shrink() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
