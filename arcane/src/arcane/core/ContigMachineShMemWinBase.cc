// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ContigMachineShMemWinBase.cc                                (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire partagée entre les         */
/* processus d'un même noeud.                                                */
/*---------------------------------------------------------------------------*/

#include "arcane/core/ContigMachineShMemWinBase.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/utils/NumericTypes.h"

#include "arccore/message_passing/internal/IContigMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ContigMachineShMemWinBase::
ContigMachineShMemWinBase(IParallelMng* pm, Int64 nb_elem_segment, Int32 sizeof_elem)
: m_pm_internal(pm->_internalApi())
, m_node_window_base(m_pm_internal->createContigMachineShMemWinBase(nb_elem_segment * static_cast<Int64>(sizeof_elem), sizeof_elem))
, m_sizeof_elem(sizeof_elem)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> ContigMachineShMemWinBase::
segmentView()
{
  return m_node_window_base->segmentView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> ContigMachineShMemWinBase::
segmentView(Int32 rank)
{
  return m_node_window_base->segmentView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> ContigMachineShMemWinBase::
windowView()
{
  return m_node_window_base->windowView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> ContigMachineShMemWinBase::
segmentConstView() const
{
  return m_node_window_base->segmentConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> ContigMachineShMemWinBase::
segmentConstView(Int32 rank) const
{
  return m_node_window_base->segmentConstView(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> ContigMachineShMemWinBase::
windowConstView() const
{
  return m_node_window_base->windowConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ContigMachineShMemWinBase::
resizeSegment(Integer new_nb_elem)
{
  m_node_window_base->resizeSegment(new_nb_elem * static_cast<Int64>(m_sizeof_elem));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> ContigMachineShMemWinBase::
machineRanks() const
{
  return m_node_window_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ContigMachineShMemWinBase::
barrier() const
{
  m_node_window_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
