// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindow.h                                (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                */
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

  class Addition
  {
   public:

    explicit Addition(DynamicMachineMemoryWindow* t)
    : m_base(t)
    {
      m_base->enableAdditionPhase();
    }
    ~Addition()
    {
      m_base->disableAdditionPhase();
    }

   private:

    DynamicMachineMemoryWindow* m_base;
  };

 public:

  DynamicMachineMemoryWindow(IParallelMng* pm, Int64 nb_elem_segment)
  : m_impl(pm, nb_elem_segment, static_cast<Int32>(sizeof(Type)))
  {}
  explicit DynamicMachineMemoryWindow(IParallelMng* pm)
  : m_impl(pm, 0, static_cast<Int32>(sizeof(Type)))
  {}

 public:

  Span<Type> segmentView() const
  {
    return asSpan<Type>(m_impl.segmentView());
  }

  Span<Type> segmentView(Int32 rank) const
  {
    return asSpan<Type>(m_impl.segmentView(rank));
  }

  void add(const Type& elem) const
  {
    auto ptr_elem = reinterpret_cast<const std::byte*>(&elem);
    m_impl.add({ ptr_elem, sizeof(Type) });
  }

  void exchangeSegmentWith(Int32 rank) const
  {
    m_impl.exchangeSegmentWith(rank);
  }

  ConstArrayView<Int32> machineRanks() const
  {
    return m_impl.machineRanks();
  }

  void barrier() const
  {
    m_impl.barrier();
  }

  void enableAdditionPhase()
  {
    m_impl.enableAdditionPhase();
  }

  void disableAdditionPhase()
  {
    m_impl.disableAdditionPhase();
  }

  void reserve(Int64 new_capacity)
  {
    m_impl.reserve(new_capacity);
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
