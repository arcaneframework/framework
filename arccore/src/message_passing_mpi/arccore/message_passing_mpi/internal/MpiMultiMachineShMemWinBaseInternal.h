// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMultiMachineShMemWinBaseInternal.h                       (C) 2000-2026 */
/*                                                                           */
/* Class allowing the creation of memory windows for a compute node.         */
/* The segments of these windows are not contiguous in memory and can        */
/* be resized. A process can possess multiple segments.                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMULTIMACHINESHMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIMULTIMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*!
 * \brief Class based on MpiMachineShMemWinBaseInternal but
 * capable of managing multiple segments per process.
 *
 * Important reminders: each main window has only one segment
 * (and not one segment per process). Thus, one MPI_Win = one segment.
 *
 * A segment is identified by the rank of its original owner and by
 * an id (which is simply the segment's position in the list of local segments).
 *
 * All arrays are 1D. To access the information of one of our segments,
 * we must calculate the position of this information using
 * our machine rank and the local position of this segment.
 * infos_pos = pos_seg + rank * m_nb_segments_per_proc
 *
 * For now, it is necessary to have the same number of segments per
 * process.
 *
 * All sizes used are in bytes. \a sizeof_type is used
 * only by MPI and for verification purposes.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMultiMachineShMemWinBaseInternal
{

 public:

  explicit MpiMultiMachineShMemWinBaseInternal(SmallSpan<Int64> sizeof_segments, Int32 nb_segments_per_proc, Int32 sizeof_type, const MPI_Comm& comm_machine, Int32 comm_machine_rank, Int32 comm_machine_size, ConstArrayView<Int32> machine_ranks);

  ~MpiMultiMachineShMemWinBaseInternal();

 public:

  /*!
   * \brief Method to get the size of an element in the window.
   *
   * Non-collective call.
   *
   * \return The size of an element.
   */
  Int32 sizeofOneElem() const;

  /*!
   * \brief Method to get the ranks that own a segment
   * in the window.
   *
   * Non-collective call.
   *
   * \return A view containing the ranks' IDs.
   */
  ConstArrayView<Int32> machineRanks() const;

  /*!
   * \brief Method to wait until all processes
   * on the node call this method to continue execution.
   */
  void barrier() const;

  /*!
   * \brief Method to get a view of one of our segments.
   *
   * Non-collective call.
   *
   * \param num_seg The position (or id) of the segment.
   * \return A view.
   */
  Span<std::byte> segmentView(Int32 num_seg);

  /*!
   * \brief Method to get a view of one of the segments
   * of another process on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the process.
   * \param num_seg The local position (or id) of the segment.
   * \return A view.
   */
  Span<std::byte> segmentView(Int32 rank, Int32 num_seg);

  /*!
   * \brief Method to get a constant view of one of our segments.
   *
   * Non-collective call.
   *
   * \param num_seg The local position (or id) of the segment.
   * \return A view.
   */
  Span<const std::byte> segmentConstView(Int32 num_seg) const;

  /*!
   * \brief Method to get a constant view of one of the segments of another
   * process on the node.
   *
   * Non-collective call.
   *
   * \param rank The rank of the process.
   * \param num_seg The local position (or id) of the segment.
   * \return A view.
   */
  Span<const std::byte> segmentConstView(Int32 rank, Int32 num_seg) const;

  /*!
   * \brief Method to request the addition of elements into one of our
   * segments.
   *
   * Non-collective call
   *
   * and can be performed by multiple threads at
   * the same time (if the parameter \a num_seg is different for each thread).
   * A call to this method with the same \a num_seg before calling
   * \a executeAdd() will replace the first call.
   *
   * A call to executeAdd() is necessary after the one or more calls to this
   * method.
   * Another method \a requestX() must not be called in the meantime.
   *
   * \param num_seg The local position (or id) of the segment.
   * \param elem The elements to add.
   */
  void requestAdd(Int32 num_seg, Span<const std::byte> elem);

  /*!
   * \brief Method to execute the addition requests.
   *
   * Collective call.
   *
   * In hybrid mode, it must only be called by a single thread per
   * process.
   */
  void executeAdd();

  /*!
   * \brief Method to request the addition of elements into one of the
   * segments of the window.
   *
   * Non-collective call
   *
   * and can be performed by multiple threads at
   * the same time (if the parameter \a thread is different for each thread).
   * A call to this method with the same \a thread before calling
   * \a executeaddToAnotherSegment() will replace the first call.
   *
   * A call to executeAddToAnotherSegment() is necessary after the one or more calls to
   * this method.
   * Another method \a requestX() must not be called in the meantime.
   *
   * Two sub-domains must not add elements into the same
   * sub-domain segment.
   *
   * \param thread The thread requesting the addition. TODO Find another way than this parameter.
   * \param rank The rank of the process owning the segment to be modified.
   * \param num_seg The local position (or id) of the segment to be modified.
   * \param elem The elements to add.
   */
  void requestAddToAnotherSegment(Int32 thread, Int32 rank, Int32 num_seg, Span<const std::byte> elem);

  /*!
   * \brief Method to execute the addition requests in the
   * segments of other processes.
   *
   * Collective call.
   *
   * In hybrid mode, it must only be called by a single thread per
   * process.
   */
  void executeAddToAnotherSegment();

  /*!
   * \brief Method to request the reservation of memory space
   * for one of our segments.
   *
   * This method does nothing if \a new_capacity is less than the memory
   * space already allocated for the segment.
   *
   * MPI will reserve space with a size greater than or equal to
   * \a new_capacity.
   *
   * This method does not resize the segment; you must always go through
   * the add() methods to add elements.
   *
   * For resizing the segment, the \a resize() methods are
   * available.
   *
   * Non-collective call
   *
   * and can be performed by multiple threads at
   * the same time (if the parameter \a num_seg is different for each thread).
   * A call to this method with the same \a num_seg before calling
   * \a executeReserve() will replace the first call.
   *
   * A call to executeReserve() is necessary after the one or more calls to this
   * method.
   * Another method \a requestX() must not be called in the meantime.
   *
   * \param num_seg The local position (or id) of the segment.
   * \param new_capacity The requested new capacity.
   */
  void requestReserve(Int32 num_seg, Int64 new_capacity);

  /*!
   * \brief Method to execute the reservation requests.
   *
   * Collective call.
   *
   * In hybrid mode, it must only be called by a single thread per
   * process.
   */
  void executeReserve();

  /*!
   * \brief Method to request the resizing of one of our
   * segments.
   *
   * If the provided size is less than the current size of the segment, the
   * elements located after the provided size will be deleted.
   *
   * If the provided size is greater than the memory space reserved for the segment,
   * a realloc will be performed.
   *
   * Non-collective call
   *
   * and can be performed by multiple threads at
   * the same time (if the parameter \a num_seg is different for each thread).
   * A call to this method with the same \a num_seg before calling
   * \a executeResize() will replace the first call.
   *
   * A call to executeResize() is necessary after the one or more calls to this
   * method.
   * Another method \a requestX() must not be called in the meantime.
   *
   * \param num_seg The local position (or id) of the segment.
   * \param new_size The new size.
   */
  void requestResize(Int32 num_seg, Int64 new_size);

  /*!
   * \brief Method to execute the resizing requests.
   *
   * Collective call.
   *
   * In hybrid mode, it must only be called by a single thread per
   * process.
   */
  void executeResize();

  /*!
   * \brief Method to reduce the reserved memory space for the
   * segments to the minimum necessary.
   *
   * Collective call.
   *
   * In hybrid mode, it must only be called by a single thread per
   * process.
   */
  void executeShrink();

 private:

  /*!
   * \brief Method to request a reallocation.
   * \param owner_pos_segment The segment to reallocate.
   * \param new_capacity The new capacity.
   */
  void _requestRealloc(Int32 owner_pos_segment, Int64 new_capacity) const;

  /*!
   * \brief Method to cancel a reallocation request.
   * \param owner_pos_segment The segment not to reallocate.
   */
  void _requestRealloc(Int32 owner_pos_segment) const;
  void _executeRealloc();
  void _realloc();

  Int32 _worldToMachine(Int32 world) const;
  Int32 _machineToWorld(Int32 machine) const;

 private:

  //! Array containing all main windows.
  //!
  //! Reminder: one MPI_Win = one segment.
  //!
  //! Segment #S of sub-domain rank R:
  //! mpi_win_seg = S + R * m_nb_segments_per_proc
  UniqueArray<MPI_Win> m_all_mpi_win;
  //! Array with views on the segments. The size of the views corresponds to all
  //! the reserved memory space.
  UniqueArray<Span<std::byte>> m_reserved_part_span;

  //! Contiguous window with resize size (or -1 if
  //! resizing is not requested).
  MPI_Win m_win_need_resize;
  //! Global view on contiguous window with resize size
  //! (or -1 if resizing is not requested).
  //!
  //! Segment #S of sub-domain rank R:
  //! need_resize_seg = S + R * m_nb_segments_per_proc
  Span<Int64> m_need_resize;

  //! Contiguous window with main window sizes.
  MPI_Win m_win_actual_sizeof;
  //! Global view on contiguous window with main window sizes.
  //!
  //! Segment #S of sub-domain rank R:
  //! sizeof_used_part_seg = S + R * m_nb_segments_per_proc
  Span<Int64> m_sizeof_used_part;

  //! Contiguous window with request for segment modification from another
  //! sub-domain.
  MPI_Win m_win_target_segments;
  //! Global view on contiguous window with request for segment modification
  //! from another sub-domain.
  //!
  //! Considering that a segment owner SD wants to modify
  //! segment ST, it performs this modification:
  //!
  //! m_target_segments[ST] = SD
  //! (See method \a requestAddToAnotherSegment()).
  //!
  //! Segment #S of sub-domain rank R:
  //! target_segments_seg = S + R * m_nb_segments_per_proc
  Span<Int32> m_target_segments;

  MPI_Comm m_comm_machine;
  Int32 m_comm_machine_size = 0;
  Int32 m_comm_machine_rank = 0;

  Int32 m_sizeof_type = 0;
  Int32 m_nb_segments_per_proc = 0;

  ConstArrayView<Int32> m_machine_ranks;

  //! Array containing the addition requests.
  //! One slot per local segment
  //! (m_add_requests.size() = m_nb_segments_per_proc).
  UniqueArray<Span<const std::byte>> m_add_requests;
  bool m_add_requested = false;

  //! Array containing the resizing requests.
  //! One slot per local segment
  //! (m_resize_requests.size() = m_nb_segments_per_proc).
  UniqueArray<Int64> m_resize_requests;
  bool m_resize_requested = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
