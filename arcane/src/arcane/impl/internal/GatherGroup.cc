// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherGroup.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Classe permettant de gérer les regroupements de données sur le ou les     */
/* sous-domaines écrivains.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/GatherGroup.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GatherGroupInfo::
GatherGroupInfo(IParallelMng* pm, bool use_collective_io)
: m_pm(pm)
, m_use_collective_io(use_collective_io)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GatherGroupInfo::
~GatherGroupInfo() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GatherGroupInfo::
computeSize(Int32 nb_elem_in)
{
  if (m_is_computed) return;
  m_is_computed = true;

  if (m_use_collective_io) {
    m_writer = m_pm->_internalApi()->masterParallelIORank();
    m_nb_sender_to_writer = m_pm->_internalApi()->nbSendersToMasterParallelIO();
  }
  else {
    m_writer = m_pm->masterIORank();
    m_nb_sender_to_writer = m_pm->commSize();
  }

  ITraceMng* tm = m_pm->traceMng();

  tm->info() << "m_writer : " << m_writer << " -- m_nb_sender : " << m_nb_sender_to_writer;

  // Si séquentiel ou MPI + MPI-IO.
  if ((m_pm->commSize() == 1) || (!m_pm->isThreadImplementation() && m_use_collective_io)) {
    m_nb_elem_output = nb_elem_in;
    m_nb_writer_global = m_pm->commSize();
    return;
  }

  if (m_writer != m_pm->commRank()) {
    m_pm->send({ 1, &nb_elem_in }, m_writer);
    m_nb_elem_output = 0;
  }
  else {
    m_nb_elem_recv.resizeNoInit(m_nb_sender_to_writer - 1);

    {
      UniqueArray<Parallel::Request> requests(m_nb_sender_to_writer - 1);
      for (Int32 i = 0; i < m_nb_sender_to_writer - 1; ++i) {
        const Int32 rank = i + m_writer + 1;
        requests[i] = m_pm->recv({ 1, &m_nb_elem_recv[i] }, rank, false);
      }
      m_pm->waitAllRequests(requests);
    }

    m_nb_elem_output = nb_elem_in;
    for (const Int32 size : m_nb_elem_recv) {
      m_nb_elem_output += size;
    }
  }

  m_nb_writer_global = m_pm->reduce(MessagePassing::ReduceSum, (m_writer == m_pm->commRank()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SmallSpan<Int32> GatherGroupInfo::
nbElemRecvGatherToMasterIO()
{
  return m_nb_elem_recv.smallSpan();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GatherGroup::
GatherGroup(GatherGroupInfo* ggi)
: m_ggi(ARCANE_CHECK_POINTER(ggi))
{
  ARCANE_FATAL_IF(!ggi->isComputed(), "GatherGroupInfo is not computed");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GatherGroup::
GatherGroup() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GatherGroup::
~GatherGroup() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GatherGroup::
needGather()
{
  ARCANE_CHECK_POINTER(m_ggi);
  IParallelMng* pm = m_ggi->m_pm;
  // True si Thread ou Hybride ou MPI sans MPI-IO
  // False si Séquentiel ou MPI + MPI-IO.
  return pm->commSize() != 1 && (pm->isThreadImplementation() || !m_ggi->m_use_collective_io);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GatherGroup::
gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out)
{
  ARCANE_CHECK_POINTER(m_ggi);
  IParallelMng* pm = m_ggi->m_pm;

  // Si séquentiel ou MPI + MPI-IO.
  if ((pm->commSize() == 1) || (!pm->isThreadImplementation() && m_ggi->m_use_collective_io)) {
    out.copy(in);
    return;
  }

  const Int32 writer = m_ggi->m_writer;

  if (writer != pm->commRank()) {
    pm->send(in.constSmallView(), writer);
    return;
  }

  out.copy(in);

  const Int32 nb_sender = m_ggi->m_nb_sender_to_writer;

  SmallSpan<const Int32> nb_elem_recved = m_ggi->m_nb_elem_recv.smallSpan();

  UniqueArray<Parallel::Request> requests(nb_sender - 1);

  Int64 old_size = in.size();
  for (Int32 i = 0; i < nb_sender - 1; ++i) {
    const Int32 rank = i + writer + 1;
    const Int64 sizeof_recved = nb_elem_recved[i] * sizeof_elem;

    ArrayView<Byte> recv_elem = out.subSpan(old_size, sizeof_recved).smallView();
    requests[i] = pm->recv(recv_elem, rank, false);

    old_size += sizeof_recved;
  }
  pm->waitAllRequests(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GatherGroup::
setGatherGroupInfo(GatherGroupInfo* ggi)
{
  ARCANE_CHECK_POINTER(ggi);
  ARCANE_FATAL_IF(!ggi->isComputed(), "GatherGroupInfo is not computed");
  m_ggi = ggi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
