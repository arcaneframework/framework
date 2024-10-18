// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.h                                      (C) 2000-2024 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZER_H
#define ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Event.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/DataTypeDispatchingDataVisitor.h"

#include "arcane/impl/internal/IDataSynchronizeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;
class INumericDataInternal;
class DataSynchronizeResult;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de synchronisation de variable.
 *
 * Une instance de cette classe est créée via
 * IParallelMng::createVariableSynchronizer(). Une instance est associée
 * à un groupe d'entité. Il faut appeller la fonction compute()
 * pour calculer les infos de synchronisation.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizer
: public TraceAccessor
, public IVariableSynchronizer
{
  friend class VariableSynchronizerComputeList;
  class SyncMessage;

 public:

  VariableSynchronizer(IParallelMng* pm, const ItemGroup& group,
                       Ref<IDataSynchronizeImplementationFactory> implementation_factory);
  ~VariableSynchronizer() override;

 public:

  IParallelMng* parallelMng() override
  {
    return m_parallel_mng;
  }

  const ItemGroup& itemGroup() override { return m_item_group; }

  void compute() override;

  void changeLocalIds(Int32ConstArrayView old_to_new_ids) override;

  void synchronize(IVariable* var) override;

  void synchronize(IVariable* var, Int32ConstArrayView local_ids) override;

  void synchronize(VariableCollection vars) override;

  void synchronize(VariableCollection vars, Int32ConstArrayView local_ids) override;

  Int32ConstArrayView communicatingRanks() override;

  Int32ConstArrayView sharedItems(Int32 index) override;

  Int32ConstArrayView ghostItems(Int32 index) override;

  void synchronizeData(IData* data) override;

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

 public:

  IVariableSynchronizerMng* synchronizeMng() const { return m_variable_synchronizer_mng; }

 private:

  IParallelMng* m_parallel_mng = nullptr;
  ItemGroup m_item_group;
  Ref<DataSynchronizeInfo> m_sync_info;
  Timer* m_sync_timer = nullptr;
  bool m_is_verbose = false;
  bool m_allow_multi_sync = true;
  bool m_trace_sync = false;
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
  Ref<IDataSynchronizeImplementationFactory> m_implementation_factory;
  IVariableSynchronizerMng* m_variable_synchronizer_mng = nullptr;
  SyncMessage* m_default_message = nullptr;
  Runner* m_runner = nullptr;
  // Pour les synchronisations sur un sous-ensemble des entités
  Ref<DataSynchronizeInfo> m_partial_sync_info;
  Ref<SyncMessage> m_partial_message;
  UniqueArray<Int32> m_partial_local_ids;
  bool m_is_check_coherence = false;

 private:

  void _synchronize(IVariable* var, SyncMessage* message);
  void _synchronizeMulti(const VariableCollection& vars, SyncMessage* message);
  bool _canSynchronizeMulti(const VariableCollection& vars);
  DataSynchronizeResult _synchronize(INumericDataInternal* data, bool is_compare_sync);
  SyncMessage* _buildMessage();
  SyncMessage* _buildMessage(Ref<DataSynchronizeInfo>& sync_info);
  void _rebuildMessage(Int32ConstArrayView local_ids);
  void _sendBeginEvent(VariableSynchronizerEventArgs& args);
  void _sendEndEvent(VariableSynchronizerEventArgs& args);
  void _sendEvent(VariableSynchronizerEventArgs& args);
  void _checkCreateTimer();
  void _doSynchronize(SyncMessage* message);
  void _setCurrentDevice();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
