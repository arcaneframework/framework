// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ContigMachineShMemWin.h                                     (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of a shared memory window between the         */
/* processes of the same node.                                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_CONTIGMACHINESHMEMWIN_H
#define ARCANE_CORE_CONTIGMACHINESHMEMWIN_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ContigMachineShMemWinBase.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the creation of a shared memory window between the
 * subdomains of the same node.
 * The segments of this window will be contiguous in memory.
 *
 * \tparam Type The type of the window elements.
 */
template <class Type>
class ContigMachineShMemWin
{
 public:

  /*!
   * \brief Constructor.
   * \param pm The ParallelMng containing the node processes.
   * \param nb_elem_segment The number of elements for the segment of this subdomain.
   */
  ContigMachineShMemWin(IParallelMng* pm, Int64 nb_elem_segment)
  : m_impl(pm, nb_elem_segment, static_cast<Int32>(sizeof(Type)))
  {}

 public:

  /*!
   * \brief Method allowing retrieval of a view on our window segment
   * memory.
   *
   * \return A view.
   */
  Span<Type> segmentView()
  {
    return asSpan<Type>(m_impl.segmentView());
  }

  /*!
   * \brief Method allowing retrieval of a view on the window segment
   * memory of another subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  Span<Type> segmentView(Int32 rank)
  {
    return asSpan<Type>(m_impl.segmentView(rank));
  }

  /*!
   * \brief Method allowing retrieval of a view on the entire memory window.
   *
   * \return A view.
   */
  Span<Type> windowView()
  {
    return asSpan<Type>(m_impl.windowView());
  }

  /*!
   * \brief Method allowing retrieval of a constant view on our segment
   * memory window.
   *
   * \return A constant view.
   */
  Span<const Type> segmentConstView() const
  {
    return asSpan<const Type>(m_impl.segmentConstView());
  }

  /*!
   * \brief Method allowing retrieval of a constant view on the segment of
   * memory window of another subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A constant view.
   */
  Span<const Type> segmentConstView(Int32 rank) const
  {
    return asSpan<const Type>(m_impl.segmentConstView(rank));
  }

  /*!
   * \brief Method allowing retrieval of a constant view on the entire window
   * memory.
   *
   * \return A constant view.
   */
  Span<const Type> windowConstView() const
  {
    return asSpan<const Type>(m_impl.windowConstView());
  }

  /*!
   * \brief Method allowing resizing of the window segments.
   * Collective call.
   *
   * The total size of the window must be less than or equal to the original size.
   *
   * \param new_nb_elem The new size of our segment.
   */
  void resizeSegment(Integer new_nb_elem)
  {
    m_impl.resizeSegment(new_nb_elem);
  }

  /*!
   * \brief Method allowing retrieval of the ranks that possess a segment
   * in the window.
   *
   * The order of the processes in the returned view corresponds to the order of
   * segments in the window.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const
  {
    return m_impl.machineRanks();
  }

  /*!
   * \brief Method allowing waiting until all processes/threads
   * of the node call this method to continue execution.
   */
  void barrier() const
  {
    m_impl.barrier();
  }

 private:

  ContigMachineShMemWinBase m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
