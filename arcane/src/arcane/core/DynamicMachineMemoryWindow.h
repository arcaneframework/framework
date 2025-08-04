// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindow.h                                (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOW_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOW_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/DynamicMachineMemoryWindowBase.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type>
class DynamicMachineMemoryWindow
{

 public:

  DynamicMachineMemoryWindow(IParallelMng* pm, Int64 nb_elem_segment)
  : m_impl(pm, nb_elem_segment, static_cast<Int32>(sizeof(Type)))
  {}
  explicit DynamicMachineMemoryWindow(IParallelMng* pm)
  : m_impl(pm, 0, static_cast<Int32>(sizeof(Type)))
  {}

 public:

  Span<Type> segmentView()
  {
    return asSpan<Type>(m_impl.segmentView());
  }

  Span<Type> segmentView(Int32 rank)
  {
    return asSpan<Type>(m_impl.segmentView(rank));
  }

  Span<const Type> segmentConstView() const
  {
    return asSpan<const Type>(m_impl.segmentConstView());
  }

  Span<const Type> segmentConstView(Int32 rank) const
  {
    return asSpan<const Type>(m_impl.segmentConstView(rank));
  }

  Int32 segmentOwner() const
  {
    return m_impl.segmentOwner();
  }
  Int32 segmentOwner(Int32 rank) const
  {
    return m_impl.segmentOwner(rank);
  }

  void add(Span<const Type> elem)
  {
    const Span<const std::byte> span_bytes(reinterpret_cast<const std::byte*>(elem.data()), elem.sizeBytes());
    m_impl.add(span_bytes);
  }

  void add()
  {
    m_impl.add();
  }

  void exchangeSegmentWith(Int32 rank)
  {
    m_impl.exchangeSegmentWith(rank);
  }

  void exchangeSegmentWith()
  {
    m_impl.exchangeSegmentWith();
  }

  void resetExchanges()
  {
    m_impl.resetExchanges();
  }

  ConstArrayView<Int32> machineRanks() const
  {
    return m_impl.machineRanks();
  }

  void barrier() const
  {
    m_impl.barrier();
  }

  void reserve(Int64 new_capacity)
  {
    m_impl.reserve(new_capacity);
  }

  void reserve()
  {
    m_impl.reserve();
  }

  void resize(Int64 new_nb_elem_segment)
  {
    m_impl.resize(new_nb_elem_segment);
  }

  void resize()
  {
    m_impl.resize();
  }

  void shrink()
  {
    m_impl.shrink();
  }

 private:

  DynamicMachineMemoryWindowBase m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
