// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.h                                   (C) 2000-2023 */
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
  : public IVariableSynchronizerMngInternal
  {
    class BufferList;

   public:

    explicit InternalApi(VariableSynchronizerMng* vms);
    ~InternalApi();

   public:

    Ref<MemoryBuffer> createSynchronizeBuffer(IMemoryAllocator* allocator) override;
    void releaseSynchronizeBuffer(MemoryBuffer* v) override;

   private:

    VariableSynchronizerMng* m_synchronizer_mng = nullptr;
    BufferList* m_buffer_list = nullptr;
  };

 public:

  void initialize();

 public:

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

  void setCompareSynchronize(bool v) { m_is_compare_synchronize = v; }
  bool isCompareSynchronize() const { return m_is_compare_synchronize; }

  void dumpStats(std::ostream& ostr) const override;
  IVariableSynchronizerMngInternal* _internalApi() { return &m_internal_api; }

 private:

  IVariableMng* m_variable_mng = nullptr;
  InternalApi m_internal_api{ this };
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
  VariableSynchronizerStats* m_stats = nullptr;
  bool m_is_compare_synchronize = false;
  bool m_is_do_stats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
