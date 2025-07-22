// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineMemoryWindow.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire partagée entre les         */
/* processus d'un même noeud.                                                */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineMemoryWindow.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/utils/NumericTypes.h"

#include "arccore/message_passing/internal/IMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
MachineMemoryWindow<Type>::
MachineMemoryWindow(IParallelMng* pm, Int64 nb_elem_segment)
: m_pm_internal(pm->_internalApi())
{
  m_node_window_base = m_pm_internal->createMachineMemoryWindowBase(nb_elem_segment * static_cast<Int64>(sizeof(Type)), static_cast<Int32>(sizeof(Type)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<Type> MachineMemoryWindow<Type>::
segmentView() const
{
  return asSpan<Type>(m_node_window_base->segment());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<Type> MachineMemoryWindow<Type>::
segmentView(Int32 rank) const
{
  return asSpan<Type>(m_node_window_base->segment(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<Type> MachineMemoryWindow<Type>::
windowView() const
{
  return asSpan<Type>(m_node_window_base->window());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<const Type> MachineMemoryWindow<Type>::
segmentConstView() const
{
  return asSpan<const Type>(m_node_window_base->segment());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<const Type> MachineMemoryWindow<Type>::
segmentConstView(Int32 rank) const
{
  return asSpan<const Type>(m_node_window_base->segment(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Span<const Type> MachineMemoryWindow<Type>::
windowConstView() const
{
  return asSpan<const Type>(m_node_window_base->window());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void MachineMemoryWindow<Type>::
resizeSegment(Integer new_nb_elem) const
{
  m_node_window_base->resizeSegment(new_nb_elem * static_cast<Int64>(sizeof(Type)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ConstArrayView<Int32> MachineMemoryWindow<Type>::
machineRanks() const
{
  return m_node_window_base->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void MachineMemoryWindow<Type>::
barrier() const
{
  m_node_window_base->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(MachineMemoryWindow);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
