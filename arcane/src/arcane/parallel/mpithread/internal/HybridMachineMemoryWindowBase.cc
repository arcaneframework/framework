// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBase.cc                            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBase.h"

#include "arcane/parallel/mpithread/HybridMessageQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineMemoryWindowBase::
HybridMachineMemoryWindowBase(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, Integer sizeof_type, Ref<IMachineMemoryWindowBase> nb_elem, Ref<IMachineMemoryWindowBase> sum_nb_elem, Ref<IMachineMemoryWindowBase> mpi_window)
: m_my_rank_local_proc(my_rank_local_proc)
, m_nb_rank_local_proc(nb_rank_local_proc)
, m_my_rank_mpi(my_rank_mpi)
, m_sizeof_type(sizeof_type)
, m_mpi_window(mpi_window)
, m_nb_elem_global(nb_elem)
, m_sum_nb_elem_global(sum_nb_elem)
, m_nb_elem_local_proc(nullptr)
, m_sum_nb_elem_local_proc(nullptr)
{
  m_nb_elem_local_proc = static_cast<Int32*>(m_nb_elem_global->data());
  m_sum_nb_elem_local_proc = static_cast<Int32*>(m_sum_nb_elem_global->data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer HybridMachineMemoryWindowBase::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer HybridMachineMemoryWindowBase::
sizeSegment() const
{
  return m_nb_elem_local_proc[m_my_rank_local_proc];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer HybridMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);

  // Si le rang est un thread de notre processus.
  if (fri.mpiRankValue() == m_my_rank_mpi) {
    return m_nb_elem_local_proc[fri.localRankValue()];
  }

  Int32* nb_elem_other_proc = static_cast<Int32*>(m_nb_elem_global->data(fri.mpiRankValue()));
  return nb_elem_other_proc[fri.localRankValue()];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* HybridMachineMemoryWindowBase::
data() const
{
  std::byte* byte_array = static_cast<std::byte*>(m_mpi_window->data());

  return &byte_array[m_sum_nb_elem_local_proc[m_my_rank_local_proc] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* HybridMachineMemoryWindowBase::
data(Int32 rank) const
{
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);

  if (fri.mpiRankValue() == m_my_rank_mpi) {
    std::byte* byte_array = static_cast<std::byte*>(m_mpi_window->data());
    return &byte_array[m_sum_nb_elem_local_proc[fri.localRankValue()] * m_sizeof_type];
  }


  Int32* sum_nb_elem_other_proc = static_cast<Int32*>(m_sum_nb_elem_global->data(fri.mpiRankValue()));

  std::byte* byte_array = static_cast<std::byte*>(m_mpi_window->data(fri.mpiRankValue()));

  return &byte_array[sum_nb_elem_other_proc[fri.localRankValue()] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> HybridMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return { sizeSegment(), data() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> HybridMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  return { sizeSegment(rank), data(rank) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
