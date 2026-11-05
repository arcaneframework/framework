// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Int64AllToAll.h                                             (C) 2000-2026 */
/*                                                                           */
/* Variable all-to-all of Int64, in rounds past what MPI counts.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INTERNAL_INT64ALLTOALL_H
#define ARCANE_MESH_INTERNAL_INT64ALLTOALL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief A variable all-to-all of Int64.
 *
 * The send buffer is grouped by destination. In one round the exchange is
 * posted non-blocking when the parallel manager offers it, so that the caller
 * can work until wait(); in several rounds it is done at once. The send buffer
 * must stay alive until wait().
 */
class Int64AllToAll
{
 public:

  /*!
   * \brief Exchanges the counts, per rank, of what an exchange will bring.
   *
   * \return the number of rounds the exchange needs so that no rank sends or
   * receives, in one round, more Int64 than MPI counts in Int32.
   */
  static Int64 exchangeCounts(IParallelMng* pm, ConstArrayView<Int64> send_count, Array<Int64>& recv_count);

  //! Posts the exchange of \a send, \a send_count[r] Int64 for rank r
  void start(IParallelMng* pm, ConstArrayView<Int64> send, ConstArrayView<Int64> send_count,
             ConstArrayView<Int64> recv_count, Int64 nb_round);

  //! Completes the exchange
  void wait();

  //! What was received, grouped by sending rank
  Span<const Int64> received() const { return m_recv.constSpan(); }

  //! What \a rank sent
  Span<const Int64> from(Int32 rank) const;

 private:

  static Int64 _nbRound(Int64 g_max, Int32 nb_rank);

 private:

  IParallelMng* m_pm = nullptr;
  UniqueArray<Int64> m_recv;
  UniqueArray<Int64> m_recv_index;
  Parallel::Request m_request;
  bool m_is_pending = false;
  Int32UniqueArray m_send_count;
  Int32UniqueArray m_send_index;
  Int32UniqueArray m_recv_count;
  Int32UniqueArray m_recv_index32;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
