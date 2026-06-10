// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IContigMachineShMemWinBaseInternal.h                        (C) 2000-2026 */
/*                                                                           */
/* Class interface allowing the creation of a memory window for a node       */
/* of computation. This window will be contiguous in memory.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_ICONTIGMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_ICONTIGMACHINESHMEMWINBASEINTERNAL_H
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
 * \brief Class allowing the creation of a memory window for a node
 * of computation.
 *
 * This window will be contiguous in memory and will be accessible by
 * all processes of the node.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IContigMachineShMemWinBaseInternal
{
 public:

  virtual ~IContigMachineShMemWinBaseInternal() = default;

 public:

  /*!
   * \brief Method allowing the retrieval of the size of an element in the window.
   *
   * \return The size of an element.
   */
  virtual Int32 sizeofOneElem() const = 0;

  /*!
   * \brief Method allowing the retrieval of a view of its segment.
   *
   * \return A view.
   */
  virtual Span<std::byte> segmentView() = 0;

  /*!
   * \brief Method allowing the retrieval of a view of the segment of another
   * subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  virtual Span<std::byte> segmentView(Int32 rank) = 0;

  /*!
   * \brief Method allowing the retrieval of a view of the entire window.
   *
   * \return A view.
   */
  virtual Span<std::byte> windowView() = 0;

  /*!
   * \brief Method allowing the retrieval of a view of its segment.
   *
   * \return A view.
   */
  virtual Span<const std::byte> segmentConstView() const = 0;

  /*!
   * \brief Method allowing the retrieval of a view of the segment of another
   * subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  virtual Span<const std::byte> segmentConstView(Int32 rank) const = 0;

  /*!
   * \brief Method allowing the retrieval of a view of the entire window.
   *
   * \return A view.
   */
  virtual Span<const std::byte> windowConstView() const = 0;

  /*!
   * \brief Method allowing the resizing of the window segments.
   *
   * Collective call.
   *
   * The total size of the window must be less than or equal to the original size.
   *
   * \param new_sizeof_segment The new size of our segment (in bytes).
   */
  virtual void resizeSegment(Int64 new_sizeof_segment) = 0;

  /*!
   * \brief Method allowing the retrieval of the ranks that possess a segment
   * in the window.
   *
   * The order of the processes in the returned view corresponds to the order of the
   * segments in the window.
   *
   * \return A view containing the rank IDs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;

  /*!
   * \brief Method allowing waiting until all processes/threads
   * of the node call this method to continue execution.
   */
  virtual void barrier() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
