// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.h                                   (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des synchroniseurs de variables.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERMNG_H
#define ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Event.h"

#include "arcane/core/IVariableSynchronizerMng.h"
#include "arcane/core/internal/IVariableSynchronizerMngInternal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableSynchronizerStats;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des synchroniseurs de variables.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerMng
: public TraceAccessor
, public IVariableSynchronizerMng
{
 public:

  explicit VariableSynchronizerMng(IVariableMng* vm);
  ~VariableSynchronizerMng();

 public:

  class InternalApi
  : public TraceAccessor
  , public IVariableSynchronizerMngInternal
  {
    class BufferList;

   public:

    explicit InternalApi(VariableSynchronizerMng* vms);
    ~InternalApi();

   public:

    Ref<MemoryBuffer> createSynchronizeBuffer(IMemoryAllocator* allocator) override;
    void releaseSynchronizeBuffer(IMemoryAllocator* allocator, MemoryBuffer* v) override;

   public:

    void dumpStats(std::ostream& ostr) const;

   private:

    VariableSynchronizerMng* m_synchronizer_mng = nullptr;
    std::unique_ptr<BufferList> m_buffer_list;
  };

 public:

  void initialize();

 public:

  IParallelMng* parallelMng() const override { return m_parallel_mng; }

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

  void setSynchronizationCompareLevel(Int32 v) final { m_synchronize_compare_level = v; }
  Int32 synchronizationCompareLevel() const final { return m_synchronize_compare_level; }
  bool isSynchronizationComparisonEnabled() const final { return m_synchronize_compare_level > 0; }

  void dumpStats(std::ostream& ostr) const override;
  void flushPendingStats() override;
  IVariableSynchronizerMngInternal* _internalApi() override { return &m_internal_api; }
  bool isDoingStats() const { return m_is_doing_stats || m_synchronize_compare_level > 0; }

 private:

  IVariableMng* m_variable_mng = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  InternalApi m_internal_api{ this };
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
  VariableSynchronizerStats* m_stats = nullptr;
  Int32 m_synchronize_compare_level = 0;
  bool m_is_doing_stats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
