// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowBase.h                            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour un noeud de calcul. */
/* Les segments de ces fenêtres ne sont pas contigües en mémoire et peuvent  */
/* être redimensionnées.                                                     */
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
  class IDynamicMachineMemoryWindowBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DynamicMachineMemoryWindowBase
{
#ifdef ARCANE_DEBUG
 private:

  template <class T>
  class Ref
  {
   public:

    explicit Ref(const Arcane::Ref<T>& ptr)
    : m_ptr(ptr)
    {}
    ~Ref() = default;

   public:

    T* operator->()
    {
      return m_ptr.get();
    }
    T const* operator->() const
    {
      return m_ptr.get();
    }

   private:

    Arcane::Ref<T> m_ptr;
  };
#endif

 public:

  DynamicMachineMemoryWindowBase(IParallelMng* pm, Int64 sizeof_segment, Int32 sizeof_elem);

 public:

  Span<std::byte> segmentView();
  Span<std::byte> segmentView(Int32 rank);

  Span<const std::byte> segmentConstView() const;
  Span<const std::byte> segmentConstView(Int32 rank) const;

  void add(Span<const std::byte> elem);
  void add();

  void addToAnotherSegment(Int32 rank, Span<const std::byte> elem);
  void addToAnotherSegment();

  ConstArrayView<Int32> machineRanks() const;

  void barrier() const;

  void reserve(Int64 new_nb_elem_segment_capacity);
  void reserve();

  void resize(Int64 new_nb_elem_segment);
  void resize();

  void shrink();

 private:

  IParallelMngInternal* m_pm_internal;
  Ref<MessagePassing::IDynamicMachineMemoryWindowBaseInternal> m_node_window_base;
  Int32 m_sizeof_elem;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
