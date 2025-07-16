// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBase.cc                      (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBase::
SharedMemoryMachineMemoryWindowBase(Int32 my_rank, Int32 nb_rank, Integer sizeof_type, std::byte* window, Integer* nb_elem, Integer* sum_nb_elem, Integer nb_elem_total)
: m_my_rank(my_rank)
, m_nb_rank(nb_rank)
, m_sizeof_type(sizeof_type)
, m_nb_elem_total(nb_elem_total)
, m_window(window)
, m_nb_elem(nb_elem)
, m_sum_nb_elem(sum_nb_elem)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBase::
~SharedMemoryMachineMemoryWindowBase()
{
  if (m_my_rank == 0) {
    delete[] m_window;
    delete[] m_nb_elem;
    delete[] m_sum_nb_elem;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeSegment() const
{
  return m_nb_elem[m_my_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  return m_nb_elem[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* SharedMemoryMachineMemoryWindowBase::
data() const
{
  return &m_window[m_sum_nb_elem[m_my_rank] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* SharedMemoryMachineMemoryWindowBase::
data(Int32 rank) const
{
  return &m_window[m_sum_nb_elem[rank] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> SharedMemoryMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return { sizeSegment(), data() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> SharedMemoryMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  return { sizeSegment(rank), data(rank) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
