// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinBase.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinBase.h"

#include "arcane/utils/NumericTypes.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arccore/message_passing/internal/IMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinBase::
MachineShMemWinBase(IParallelMng* pm, Int64 nb_elem_segment, Int32 sizeof_elem)
: m_pm_internal(pm->_internalApi())
, m_node_window_base(m_pm_internal->createMachineShMemWinBase(nb_elem_segment * static_cast<Int64>(sizeof_elem), sizeof_elem))
, m_sizeof_elem(sizeof_elem)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineShMemWinBase::
machineRanks() const
{
  return m_node_window_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
barrier() const
{
  m_node_window_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinBase::
segmentView()
{
  return m_node_window_base->segmentView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinBase::
segmentView(Int32 rank)
{
  return m_node_window_base->segmentView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MachineShMemWinBase::
segmentConstView() const
{
  return m_node_window_base->segmentConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> MachineShMemWinBase::
segmentConstView(Int32 rank) const
{
  return m_node_window_base->segmentConstView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
add(Span<const std::byte> elem)
{
  return m_node_window_base->add(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
add()
{
  return m_node_window_base->add();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
addToAnotherSegment(Int32 rank, Span<const std::byte> elem)
{
  m_node_window_base->addToAnotherSegment(rank, elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
addToAnotherSegment()
{
  m_node_window_base->addToAnotherSegment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
reserve(Int64 new_nb_elem_segment_capacity)
{
  m_node_window_base->reserve(new_nb_elem_segment_capacity * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
reserve()
{
  m_node_window_base->reserve();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
resize(Int64 new_nb_elem_segment)
{
  m_node_window_base->resize(new_nb_elem_segment * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
resize()
{
  m_node_window_base->resize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinBase::
shrink()
{
  m_node_window_base->shrink();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
