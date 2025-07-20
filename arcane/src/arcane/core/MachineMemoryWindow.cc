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

#include "arccore/message_passing/IMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
MachineMemoryWindow<Type>::
MachineMemoryWindow(IParallelMng* pm, Integer nb_elem_local)
: m_pm_internal(pm->_internalApi())
{
  m_node_window_base = m_pm_internal->createMachineMemoryWindowBase(nb_elem_local, sizeof(Type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Integer MachineMemoryWindow<Type>::
sizeSegment() const
{
  return m_node_window_base->sizeSegment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Integer MachineMemoryWindow<Type>::
sizeSegment(Int32 rank) const
{
  return m_node_window_base->sizeSegment(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Integer MachineMemoryWindow<Type>::
sizeWindow() const
{
  return m_node_window_base->sizeWindow();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ArrayView<Type> MachineMemoryWindow<Type>::
segmentView() const
{
  auto [size, data] = m_node_window_base->sizeAndDataSegment();
  return ArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ArrayView<Type> MachineMemoryWindow<Type>::
segmentView(Int32 rank) const
{
  auto [size, data] = m_node_window_base->sizeAndDataSegment(rank);
  return ArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ArrayView<Type> MachineMemoryWindow<Type>::
windowView() const
{
  auto [size, data] = m_node_window_base->sizeAndDataWindow();
  return ArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ConstArrayView<Type> MachineMemoryWindow<Type>::
segmentConstView() const
{
  auto [size, data] = m_node_window_base->sizeAndDataSegment();
  return ConstArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ConstArrayView<Type> MachineMemoryWindow<Type>::
segmentConstView(Int32 rank) const
{
  auto [size, data] = m_node_window_base->sizeAndDataSegment(rank);
  return ConstArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
ConstArrayView<Type> MachineMemoryWindow<Type>::
windowConstView() const
{
  auto [size, data] = m_node_window_base->sizeAndDataWindow();
  return ConstArrayView<Type>(size, static_cast<Type*>(data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Type* MachineMemoryWindow<Type>::
dataSegment() const
{
  return static_cast<Type*>(m_node_window_base->dataSegment());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Type* MachineMemoryWindow<Type>::
dataSegment(Int32 rank) const
{
  return static_cast<Type*>(m_node_window_base->dataSegment(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
Type* MachineMemoryWindow<Type>::
dataWindow() const
{
  return static_cast<Type*>(m_node_window_base->dataWindow());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
void MachineMemoryWindow<Type>::
resizeSegment(Integer new_nb_elem) const
{
  m_node_window_base->resizeSegment(new_nb_elem);
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

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(MachineMemoryWindow);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
