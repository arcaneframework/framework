// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowBase.cc                           (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#include "arcane/core/DynamicMachineMemoryWindowBase.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/FatalErrorException.h"

#include "arccore/message_passing/internal/IDynamicMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMachineMemoryWindowBase::
DynamicMachineMemoryWindowBase(IParallelMng* pm, Int64 nb_elem_segment, Int32 sizeof_elem)
: m_pm_internal(pm->_internalApi())
, m_sizeof_elem(sizeof_elem)
{
  m_node_window_base = m_pm_internal->createDynamicMachineMemoryWindowBase(nb_elem_segment * static_cast<Int64>(m_sizeof_elem), m_sizeof_elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowBase::
segmentView() const
{
  return m_node_window_base->segment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowBase::
segmentView(Int32 rank) const
{
  return m_node_window_base->segment(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 DynamicMachineMemoryWindowBase::
segmentOwner() const
{
  return m_node_window_base->segmentOwner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 DynamicMachineMemoryWindowBase::
segmentOwner(Int32 rank) const
{
  return m_node_window_base->segmentOwner(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
add(Span<const std::byte> elem) const
{
  return m_node_window_base->add(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
add() const
{
  return m_node_window_base->add();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
exchangeSegmentWith(Int32 rank) const
{
  m_node_window_base->exchangeSegmentWith(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
exchangeSegmentWith() const
{
  m_node_window_base->exchangeSegmentWith();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
resetExchanges() const
{
  m_node_window_base->resetExchanges();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> DynamicMachineMemoryWindowBase::
machineRanks() const
{
  return m_node_window_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
barrier() const
{
  m_node_window_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
reserve(Int64 new_nb_elem_segment_capacity) const
{
  m_node_window_base->reserve(new_nb_elem_segment_capacity * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
reserve() const
{
  m_node_window_base->reserve();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
resize(Int64 new_nb_elem_segment) const
{
  m_node_window_base->resize(new_nb_elem_segment * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
resize() const
{
  m_node_window_base->resize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowBase::
shrink() const
{
  m_node_window_base->shrink();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
