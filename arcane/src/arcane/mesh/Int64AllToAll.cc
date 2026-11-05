// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Int64AllToAll.cc                                            (C) 2000-2026 */
/*                                                                           */
/* Variable all-to-all of Int64, in rounds past what MPI counts.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/internal/Int64AllToAll.h"

#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelNonBlockingCollective.h"

#include <algorithm>
#include <cstdint>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \a g_max is the largest number of Int64 any rank sends or receives over the
 * whole exchange. A round sends at most ceil(count / nb_round) to each rank, so
 * at most g_max / nb_round + nb_rank in all.
 */
Int64 Int64AllToAll::
_nbRound(Int64 g_max, Int32 nb_rank)
{
  const Int64 room = static_cast<Int64>(INT32_MAX) - nb_rank;
  return std::max<Int64>(1, (g_max + room - 1) / room);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 Int64AllToAll::
exchangeCounts(IParallelMng* pm, ConstArrayView<Int64> send_count, Array<Int64>& recv_count)
{
  const Int32 nb_rank = pm->commSize();
  recv_count.resize(nb_rank);
  pm->allToAll(send_count, recv_count, 1);
  Int64 total_send = 0;
  Int64 total_recv = 0;
  for (Int32 r = 0; r < nb_rank; ++r) {
    total_send += send_count[r];
    total_recv += recv_count[r];
  }
  return _nbRound(pm->reduce(Parallel::ReduceMax, std::max(total_send, total_recv)), nb_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Int64AllToAll::
start(IParallelMng* pm, ConstArrayView<Int64> send, ConstArrayView<Int64> send_count,
      ConstArrayView<Int64> recv_count, Int64 nb_round)
{
  m_pm = pm;
  const Int32 nb_rank = pm->commSize();
  m_recv_index.resize(nb_rank + 1);
  m_recv_index[0] = 0;
  for (Int32 r = 0; r < nb_rank; ++r)
    m_recv_index[r + 1] = m_recv_index[r] + recv_count[r];
  m_recv.resize(m_recv_index[nb_rank]);
  UniqueArray<Int64> send_index(nb_rank + 1);
  send_index[0] = 0;
  for (Int32 r = 0; r < nb_rank; ++r)
    send_index[r + 1] = send_index[r] + send_count[r];

  m_send_count.resize(nb_rank);
  m_send_index.resize(nb_rank);
  m_recv_count.resize(nb_rank);
  m_recv_index32.resize(nb_rank);
  IParallelNonBlockingCollective* nbc = pm->nonBlockingCollective();
  if (nb_round == 1) {
    for (Int32 r = 0; r < nb_rank; ++r) {
      m_send_count[r] = CheckedConvert::toInt32(send_count[r]);
      m_send_index[r] = CheckedConvert::toInt32(send_index[r]);
      m_recv_count[r] = CheckedConvert::toInt32(recv_count[r]);
      m_recv_index32[r] = CheckedConvert::toInt32(m_recv_index[r]);
    }
    if (nbc) {
      m_request = nbc->allToAllVariable(send, m_send_count, m_send_index, m_recv, m_recv_count, m_recv_index32);
      m_is_pending = true;
    }
    else
      pm->allToAllVariable(send, m_send_count, m_send_index, m_recv, m_recv_count, m_recv_index32);
    return;
  }

  // Round k carries [count * k / nb_round, count * (k+1) / nb_round) of every
  // message; both sides compute the same cut from the same count.
  UniqueArray<Int64> sbuf;
  UniqueArray<Int64> rbuf;
  for (Int64 k = 0; k < nb_round; ++k) {
    Int32 s = 0;
    Int32 t = 0;
    for (Int32 r = 0; r < nb_rank; ++r) {
      const Int64 slo = send_count[r] * k / nb_round;
      const Int64 shi = send_count[r] * (k + 1) / nb_round;
      m_send_count[r] = CheckedConvert::toInt32(shi - slo);
      m_send_index[r] = s;
      s += m_send_count[r];
      const Int64 rlo = recv_count[r] * k / nb_round;
      const Int64 rhi = recv_count[r] * (k + 1) / nb_round;
      m_recv_count[r] = CheckedConvert::toInt32(rhi - rlo);
      m_recv_index32[r] = t;
      t += m_recv_count[r];
    }
    sbuf.resize(s);
    for (Int32 r = 0; r < nb_rank; ++r) {
      const Int64 slo = send_index[r] + send_count[r] * k / nb_round;
      for (Int32 z = 0; z < m_send_count[r]; ++z)
        sbuf[m_send_index[r] + z] = send.data()[slo + z];
    }
    rbuf.resize(t);
    pm->allToAllVariable(sbuf, m_send_count, m_send_index, rbuf, m_recv_count, m_recv_index32);
    for (Int32 r = 0; r < nb_rank; ++r) {
      const Int64 rlo = m_recv_index[r] + recv_count[r] * k / nb_round;
      for (Int32 z = 0; z < m_recv_count[r]; ++z)
        m_recv[rlo + z] = rbuf[m_recv_index32[r] + z];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Int64AllToAll::
wait()
{
  if (m_is_pending)
    m_pm->waitAllRequests(ArrayView<Parallel::Request>(1, &m_request));
  m_is_pending = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Int64> Int64AllToAll::
from(Int32 rank) const
{
  return m_recv.constSpan().subSpan(m_recv_index[rank], m_recv_index[rank + 1] - m_recv_index[rank]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
