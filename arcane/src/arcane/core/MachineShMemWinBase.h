// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinBase.h                                       (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of memory windows for a computing node.       */
/* The segments of these windows are not contiguous in memory and can        */
/* be resized.                                                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINESHMEMWINBASE_H
#define ARCANE_CORE_MACHINESHMEMWINBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IParallelMngInternal;
namespace MessagePassing
{
  class IMachineShMemWinBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the creation of a shared memory window between the
 * subdomains of the same node.
 *
 * The segments of this window are not contiguous in memory and can
 * be resized.
 *
 * The \a add() method allows elements to be added iteratively.
 */
class ARCANE_CORE_EXPORT MachineShMemWinBase
{

 public:

  /*!
   * \brief Constructor.
   * \param pm The parallelMng to use.
   * \param sizeof_segment The total size of our segment (in bytes / must
   * be divisible by \a sizeof_elem).
   * \param sizeof_elem The size of an element in our segment (in bytes).
   */
  MachineShMemWinBase(IParallelMng* pm, Int64 sizeof_segment, Int32 sizeof_elem);

 public:

  /*!
   * \brief Method to obtain the ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method to wait until all processes/threads
   * on the node call this method to continue execution.
   */
  void barrier() const;

  /*!
   * \brief Method to obtain a view of our segment.
   *
   * Non-collective call.
   *
   * \return A view.
   */
  Span<std::byte> segmentView();

  /*!
   * \brief Method to obtain a view of the segment of another
   * subdomain on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  Span<std::byte> segmentView(Int32 rank);

  /*!
   * \brief Method to obtain a view of our segment.
   *
   * Non-collective call.
   *
   * \return A view.
   */
  Span<const std::byte> segmentConstView() const;

  /*!
   * \brief Method to obtain a view of the segment of another
   * subdomain on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  Span<const std::byte> segmentConstView(Int32 rank) const;

  /*!
   * \brief Method to add elements into our segment.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * If the segment is too small, it will be resized.
   *
   * Subdomains that do not wish to add elements can call the
   * \a add() method without parameters or this method with an empty view.
   *
   * \param elem The elements to add.
   */
  void add(Span<const std::byte> elem);
  /*!
   * \brief Method to be called by the subdomain(s) that do not wish to add
   * elements to its segment.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * See the documentation for \a add(Span<const std::byte> elem).
   */
  void add();

  /*!
   * \brief Method to add elements into the segment of another
   * subdomain.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * Two subdomains must not add elements to the same subdomain segment.
   *
   * If the target segment is too small, it will be resized.
   *
   * Subdomains that do not wish to add elements can call the
   * \a addToAnotherSegment() method without parameters.
   *
   * \param rank The rank of the subdomain whose segment is to be modified.
   * \param elem The elements to add.
   */
  void addToAnotherSegment(Int32 rank, Span<const std::byte> elem);

  /*!
   * \brief Method to be called by the subdomain(s) that do not wish to add
   * elements into the segment of another subdomain.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * See the documentation for \a addToAnotherSegment(Int32 rank, Span<const Type> elem).
   */
  void addToAnotherSegment();

  /*!
   * \brief Method to reserve memory space in our segment.
   *
   * Collective call.
   *
   * This method does nothing if \a new_capacity is less than the memory space
   * already allocated for the segment.
   * For subdomains that do not wish to increase the available memory space
   * for their segment, it is possible to set the \a new_capacity parameter
   * to 0 or use the \a reserve() method (without arguments).
   *
   * The reserved space will have a size greater than or equal to
   * \a new_capacity.
   *
   * This method does not resize the segment; you must always use the add()
   * method to add elements.
   *
   * To resize the segment, the \a resize(Int64 new_size) method is
   * available.
   *
   * \param new_nb_elem_segment_capacity The requested new capacity (in number of elements, not in bytes).
   */
  void reserve(Int64 new_nb_elem_segment_capacity);

  /*!
   * \brief Method to be called by the subdomain(s) that do not wish to reserve
   * more memory for their segments.
   *
   * Collective call.
   *
   * See the documentation for \a reserve(Int64 new_nb_elem_segment_capacity).
   */
  void reserve();

  /*!
   * \brief Method to resize our segment.
   *
   * Collective call.
   *
   * If the provided size is less than the current size of the segment, elements
   * located after the provided size will be deleted.
   *
   * For subdomains that do not wish to resize their segment, it is
   * possible to set the \a new_size argument to -1 or call the method
   * \a resize() (without arguments).
   *
   * \param new_nb_elem_segment The new size (in number of elements, not in bytes).
   */
  void resize(Int64 new_nb_elem_segment);

  /*!
   * \brief Method to be called by the subdomain(s) that do not wish to
   * resize their segments.
   *
   * Collective call.
   *
   * See the documentation for \a resize(Int64 new_nb_elem_segment).
   */
  void resize();

  /*!
   * \brief Method to reduce the reserved memory space for the
   * segments to the minimum necessary.
   *
   * Collective call.
   */
  void shrink();

 private:

  IParallelMngInternal* m_pm_internal;
  Ref<MessagePassing::IMachineShMemWinBaseInternal> m_node_window_base;
  Int32 m_sizeof_elem;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
