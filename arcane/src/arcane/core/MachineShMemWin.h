// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWin.h                                           (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of memory windows for a compute node.         */
/* The segments of these windows are not contiguous in memory and can        */
/* be resized.                                                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_MACHINESHMEMWIN_H
#define ARCANE_CORE_MACHINESHMEMWIN_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinBase.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the creation of a shared memory window between
 * sub-domains of the same node.
 *
 * The segments of this window are not contiguous in memory and can
 * be resized.
 *
 * The \a add() method allows adding elements iteratively.
 *
 * \tparam Type The type of the window elements.
 */
template <class Type>
class MachineShMemWin
{

 public:

  /*!
   * \brief Constructor.
   * \param pm The parallelMng to use.
   * \param nb_elem_segment The initial number of elements.
   */
  MachineShMemWin(IParallelMng* pm, Int64 nb_elem_segment)
  : m_impl(pm, nb_elem_segment, static_cast<Int32>(sizeof(Type)))
  {}

  /*!
   * \brief Constructor.
   * \param pm The parallelMng to use.
   */
  explicit MachineShMemWin(IParallelMng* pm)
  : m_impl(pm, 0, static_cast<Int32>(sizeof(Type)))
  {}

 public:

  /*!
   * \brief Method to obtain the ranks that possess a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const
  {
    return m_impl.machineRanks();
  }

  /*!
   * \brief Method to wait until all processes/threads
   * on the node call this method to continue execution.
   */
  void barrier() const
  {
    m_impl.barrier();
  }
  /*!
   * \brief Method to obtain a view of our segment.
   *
   * Non-collective call.
   *
   * \return A view.
   */
  Span<Type> segmentView()
  {
    return asSpan<Type>(m_impl.segmentView());
  }

  /*!
   * \brief Method to obtain a view of the segment of another
   * sub-domain on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the sub-domain.
   * \return A view.
   */
  Span<Type> segmentView(Int32 rank)
  {
    return asSpan<Type>(m_impl.segmentView(rank));
  }

  /*!
   * \brief Method to obtain a view of our segment.
   *
   * Non-collective call.
   *
   * \return A view.
   */
  Span<const Type> segmentConstView() const
  {
    return asSpan<const Type>(m_impl.segmentConstView());
  }

  /*!
   * \brief Method to obtain a view of the segment of another
   * sub-domain on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the sub-domain.
   * \return A view.
   */
  Span<const Type> segmentConstView(Int32 rank) const
  {
    return asSpan<const Type>(m_impl.segmentConstView(rank));
  }

  /*!
   * \brief Method to add elements to our segment.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * If the segment is too small, it will be resized.
   *
   * Sub-domains that do not wish to add elements can call the
   * \a add() method without parameters or this method with an empty view.
   *
   * \param elem The elements to add.
   */
  void add(Span<const Type> elem)
  {
    const Span<const std::byte> span_bytes(reinterpret_cast<const std::byte*>(elem.data()), elem.sizeBytes());
    m_impl.add(span_bytes);
  }

  /*!
   * \brief Method to be called by the sub-domain(s) that do not wish to add
   * elements to their segment.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * See the documentation for \a add(Span<const std::byte> elem).
   */
  void add()
  {
    m_impl.add();
  }

  /*!
   * \brief Method to add elements to the segment of another
   * sub-domain.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * Two sub-domains must not add elements to the same
   * sub-domain segment.
   *
   * If the target segment is too small, it will be resized.
   *
   * Sub-domains that do not wish to add elements can call the
   * \a addToAnotherSegment() method without parameters.
   *
   * \param rank The rank of the sub-domain with the segment to modify.
   * \param elem The elements to add.
   */
  void addToAnotherSegment(Int32 rank, Span<const Type> elem)
  {
    const Span<const std::byte> span_bytes(reinterpret_cast<const std::byte*>(elem.data()), elem.sizeBytes());
    m_impl.addToAnotherSegment(rank, span_bytes);
  }

  /*!
   * \brief Method to be called by the sub-domain(s) that do not wish to add
   * elements to the segment of another sub-domain.
   *
   * Collective call.
   *
   * \note The methods add(..) and addToAnotherSegment(..) must not be mixed.
   *
   * See the documentation for \a addToAnotherSegment(Int32 rank, Span<const Type> elem).
   */
  void addToAnotherSegment()
  {
    m_impl.addToAnotherSegment();
  }

  /*!
   * \brief Method to reserve memory space in our segment.
   *
   * Collective call.
   *
   * This method does nothing if \a new_capacity is less than the memory
   * already allocated for the segment.
   * For sub-domains that do not wish to increase the memory space
   * available for their segment, it is possible to set the parameter
   * \a new_capacity to 0 or use the \a reserve() method (without
   * arguments).
   *
   * The reserved space will have a size greater than or equal to
   * \a new_capacity.
   *
   * This method does not resize the segment; you must always use
   * the add() method to add elements.
   *
   * To resize the segment, the \a resize(Int64 new_size) method is
   * available.
   *
   * \param new_capacity The requested new capacity.
   */
  void reserve(Int64 new_capacity)
  {
    m_impl.reserve(new_capacity);
  }

  /*!
   * \brief Method to be called by the sub-domain(s) that do not wish to reserve
   * more memory for their segments.
   *
   * Collective call.
   *
   * See the documentation for \a reserve(Int64 new_capacity).
   */
  void reserve()
  {
    m_impl.reserve();
  }

  /*!
   * \brief Method to resize our segment.
   *
   * Collective call.
   *
   * If the provided size is less than the current size of the segment,
   * elements located after the provided size will be deleted.
   *
   * For sub-domains that do not wish to resize their segment, it is
   * possible to set the argument \a new_size to -1 or call the method
   * \a resize() (without arguments).
   *
   * \param new_nb_elem The new size.
   */
  void resize(Int64 new_nb_elem)
  {
    m_impl.resize(new_nb_elem);
  }

  /*!
   * \brief Method to be called by the sub-domain(s) that do not wish to
   * resize their segments.
   *
   * Collective call.
   *
   * See the documentation for \a resize(Int64 new_nb_elem).
   */
  void resize()
  {
    m_impl.resize();
  }

  /*!
   * \brief Method to reduce the reserved memory space for the
   * segments to the minimum necessary.
   *
   * Collective call.
   */
  void shrink()
  {
    m_impl.shrink();
  }

 private:

  MachineShMemWinBase m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
