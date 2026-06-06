// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ContigMachineShMemWinBase.h                                 (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of a shared memory window between the         */
/* processes of the same node.                                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_CONTIGMACHINESHMEMWINBASE_H
#define ARCANE_CORE_CONTIGMACHINESHMEMWINBASE_H

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
  class IContigMachineShMemWinBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the creation of a shared memory window between the
 * subdomains of the same node.
 * The segments of this window will be contiguous in memory.
 */
class ARCANE_CORE_EXPORT ContigMachineShMemWinBase
{

 public:

  /*!
   * \brief Constructor.
   * \param pm The ParallelMng containing the node's processes.
   * \param sizeof_segment The size of the segment for this subdomain (in bytes).
   * \param sizeof_elem The size of an element (in bytes).
   */
  ContigMachineShMemWinBase(IParallelMng* pm, Int64 sizeof_segment, Int32 sizeof_elem);

 public:

  /*!
   * \brief Method allowing retrieval of a view on our window segment
   * memory.
   *
   * \return A view.
   */
  Span<std::byte> segmentView();

  /*!
   * \brief Method allowing retrieval of a view on the window segment
   * memory of another subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A view.
   */
  Span<std::byte> segmentView(Int32 rank);

  /*!
   * \brief Method allowing retrieval of a view on the entire memory window.
   *
   * \return A view.
   */
  Span<std::byte> windowView();

  /*!
   * \brief Method allowing retrieval of a constant view on our segment
   * memory window.
   *
   * \return A constant view.
   */
  Span<const std::byte> segmentConstView() const;

  /*!
   * \brief Method allowing retrieval of a constant view on the segment of
   * memory window of another subdomain of the node.
   *
   * \param rank The rank of the subdomain.
   * \return A constant view.
   */
  Span<const std::byte> segmentConstView(Int32 rank) const;

  /*!
   * \brief Method allowing retrieval of a constant view on the entire window
   * memory.
   *
   * \return A constant view.
   */
  Span<const std::byte> windowConstView() const;

  /*!
   * \brief Method allowing resizing of the window segments.
   * Collective call.
   *
   * The total size of the window must be less than or equal to the original size.
   *
   * \param new_size The new size of our segment (in bytes).
   */
  void resizeSegment(Integer new_size);

  /*!
   * \brief Method allowing retrieval of the ranks that possess a segment
   * in the window.
   *
   * The order of the processes in the returned view corresponds to the order of
   * segments in the window.
   *
   * \return A view containing the rank IDs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method allowing waiting until all processes/threads
   * of the node call this method to continue execution.
   */
  void barrier() const;

 private:

  IParallelMngInternal* m_pm_internal;
  Ref<MessagePassing::IContigMachineShMemWinBaseInternal> m_node_window_base;
  Int32 m_sizeof_elem;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
