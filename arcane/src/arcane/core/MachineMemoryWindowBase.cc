// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineMemoryWindowBase.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire partagée entre les         */
/* processus d'un même noeud.                                                */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineMemoryWindowBase.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/utils/NumericTypes.h"

#include "arccore/message_passing/internal/IMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineMemoryWindowBase::
MachineMemoryWindowBase(IParallelMng* pm, Int64 nb_elem_segment, Int32 sizeof_elem)
: m_pm_internal(pm->_internalApi())
, m_node_window_base(m_pm_internal->createMachineMemoryWindowBase(nb_elem_segment * static_cast<Int64>(sizeof_elem), sizeof_elem))
, m_sizeof_elem(sizeof_elem)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineMemoryWindowBase::
segmentView()
{
  return m_node_window_base->segmentView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineMemoryWindowBase::
segmentView(Int32 rank)
{
  return m_node_window_base->segmentView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineMemoryWindowBase::
windowView()
{
  return m_node_window_base->windowView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MachineMemoryWindowBase::
segmentConstView() const
{
  return m_node_window_base->segmentConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MachineMemoryWindowBase::
segmentConstView(Int32 rank) const
{
  return m_node_window_base->segmentConstView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MachineMemoryWindowBase::
windowConstView() const
{
  return m_node_window_base->windowConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineMemoryWindowBase::
resizeSegment(Integer new_nb_elem)
{
  m_node_window_base->resizeSegment(new_nb_elem * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineMemoryWindowBase::
machineRanks() const
{
  return m_node_window_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineMemoryWindowBase::
barrier() const
{
  m_node_window_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
