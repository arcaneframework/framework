// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowBase.h                            (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWBASE_H
#define ARCANE_CORE_DYNAMICMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IParallelMngInternal;
namespace MessagePassing
{
  class IDynamicMachineMemWinBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DynamicMachineMemoryWindowBase
{

 public:

  class ARCANE_CORE_EXPORT Addition
  {
   public:

    //! Associe le timer \a t et le démarre
    explicit Addition(DynamicMachineMemoryWindowBase* t)
    : m_base(t)
    {
      m_base->enableAdditionPhase();
    }

    ~Addition()
    {
      m_base->disableAdditionPhase();
    }

   private:

    DynamicMachineMemoryWindowBase* m_base; //!< Timer associé
  };

 public:

  DynamicMachineMemoryWindowBase(IParallelMng* pm, Int64 sizeof_segment, Int32 sizeof_elem);

 public:

  Span<std::byte> segmentView() const;
  Span<std::byte> segmentView(Int32 rank) const;

  void add(Span<const std::byte> elem) const;

  void exchangeSegmentWith(Int32 rank) const;

  ConstArrayView<Int32> machineRanks() const;

  void barrier() const;

  void enableAdditionPhase();
  void disableAdditionPhase();

  void reserve(Int64 new_nb_elem_segment_capacity) const;

 private:

  IParallelMngInternal* m_pm_internal;
  Ref<MessagePassing::IDynamicMachineMemWinBaseInternal> m_node_window_base;
  Int32 m_sizeof_elem;
  bool m_is_add_enabled;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
