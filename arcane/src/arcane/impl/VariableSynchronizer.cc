// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.cc                                     (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizer.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariableSynchronizerMng.h"
#include "arcane/core/parallel/IStat.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/impl/DataSynchronizeInfo.h"
#include "arcane/impl/internal/VariableSynchronizerComputeList.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm);

class VariableSynchronizer::SyncMessage
{
 public:

  SyncMessage(const DataSynchronizeDispatcherBuildInfo& bi)
  : m_dispatcher(IDataSynchronizeDispatcher::create(bi))
  , m_multi_dispatcher(IDataSynchronizeMultiDispatcher::create(bi))
  {
    if (!m_dispatcher)
      ARCANE_FATAL("No synchronizer created");
    if (!m_multi_dispatcher)
      ARCANE_FATAL("No multi synchronizer created");
  }
  ~SyncMessage()
  {
    delete m_multi_dispatcher;
  }

 public:

  void compute()
  {
    m_dispatcher->compute();
    m_multi_dispatcher->compute();
  }

  DataSynchronizeResult synchronize(INumericDataInternal* data, bool is_compare_sync)
  {
    m_dispatcher->beginSynchronize(data, is_compare_sync);
    return m_dispatcher->endSynchronize();
  }

 public:

  Ref<IDataSynchronizeDispatcher> m_dispatcher;
  IDataSynchronizeMultiDispatcher* m_multi_dispatcher = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
VariableSynchronizer(IParallelMng* pm, const ItemGroup& group,
                     Ref<IDataSynchronizeImplementationFactory> implementation_factory)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_item_group(group)
{
  m_sync_info = DataSynchronizeInfo::create();
  if (!implementation_factory.get())
    implementation_factory = arcaneCreateSimpleVariableSynchronizerFactory(pm);
  m_implementation_factory = implementation_factory;

  m_variable_synchronizer_mng = group.itemFamily()->mesh()->variableMng()->synchronizerMng();

  {
    String s = platform::getEnvironmentVariable("ARCANE_ALLOW_MULTISYNC");
    if (s == "0" || s == "FALSE" || s == "false")
      m_allow_multi_sync = false;
  }
  {
    String s = platform::getEnvironmentVariable("ARCANE_TRACE_SYNCHRONIZE");
    if (s == "1" || s == "TRUE" || s == "true")
      m_trace_sync = true;
  }

  m_default_message = _buildMessage();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
~VariableSynchronizer()
{
  delete m_sync_timer;
  delete m_default_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::SyncMessage* VariableSynchronizer::
_buildMessage()
{
  GroupIndexTable* table = nullptr;
  if (!m_item_group.isAllItems())
    table = m_item_group.localIdToIndex().get();
  DataSynchronizeDispatcherBuildInfo bi(m_parallel_mng, table, m_implementation_factory, m_sync_info);
  return new SyncMessage(bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Création de la liste des éléments de synchronisation.
 */
void VariableSynchronizer::
compute()
{
  VariableSynchronizerComputeList computer(this);
  computer.compute();

  m_default_message->compute();
  if (m_is_verbose)
    info() << "End compute dispatcher Date=" << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronize(IVariable* var)
{
  IParallelMng* pm = m_parallel_mng;
  ITimeStats* ts = pm->timeStats();
  Timer::Phase tphase(ts, TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " Sync variable " << var->fullName();
  if (m_trace_sync) {
    info() << " Synchronize variable " << var->fullName()
           << " stack=" << platform::getStackTrace();
  }
  auto& global_on_synchronized = m_variable_synchronizer_mng->onSynchronized();
  bool has_observers = global_on_synchronized.hasObservers() || m_on_synchronized.hasObservers();
  // Debut de la synchro
  if (has_observers) {
    VariableSynchronizerEventArgs args(var, this);
    global_on_synchronized.notify(args);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(), "SyncTimer", Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    _synchronize(var);
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("Synchronize", elapsed_time, 1);
  // Fin de la synchro
  if (global_on_synchronized.hasObservers() || m_on_synchronized.hasObservers()) {
    VariableSynchronizerEventArgs args(var, this, elapsed_time);
    global_on_synchronized.notify(args);
    m_on_synchronized.notify(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronize(VariableCollection vars)
{
  if (vars.empty())
    return;

  const bool use_multi = m_allow_multi_sync;
  if (use_multi && _canSynchronizeMulti(vars)) {
    _synchronizeMulti(vars);
  }
  else {
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      synchronize(*ivar);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeResult VariableSynchronizer::
_synchronize(INumericDataInternal* data, bool is_compare_sync)
{
  return m_default_message->synchronize(data, is_compare_sync);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_synchronize(IVariable* var)
{
  ARCANE_CHECK_POINTER(var);
  INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data", var->name());
  bool is_compare_sync = m_variable_synchronizer_mng->isCompareSynchronize();
  DataSynchronizeResult result = _synchronize(numapi, is_compare_sync);
  eDataSynchronizeCompareStatus s = result.compareStatus();
  if (is_compare_sync) {
    if (s == eDataSynchronizeCompareStatus::Different)
      info() << "Different values name=" << var->name();
    else if (s == eDataSynchronizeCompareStatus::Same)
      info() << "Same values name=" << var->name();
    else
      info() << "Unknown values name=" << var->name();
  }
  var->setIsSynchronized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronizeData(IData* data)
{
  ARCANE_CHECK_POINTER(data);
  INumericDataInternal* numapi = data->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Data can not be synchronized because it is not a numeric data");
  _synchronize(numapi, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  info(4) << "** VariableSynchronizer::changeLocalIds() group=" << m_item_group.name();
  m_sync_info->changeLocalIds(old_to_new_ids);
  m_default_message->compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique si les variables de la liste \a vars peuvent être synchronisées
 * en une seule fois.
 *
 * Pour que cela soit possible, il faut que ces variables ne soient pas
 * partielles et reposent sur le même ItemGroup (donc soient de la même famille)
 */
bool VariableSynchronizer::
_canSynchronizeMulti(const VariableCollection& vars)
{
  if (vars.count() == 1)
    return false;
  ItemGroup group;
  bool is_set = false;
  for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
    IVariable* var = *ivar;
    if (var->isPartial())
      return false;
    ItemGroup var_group = var->itemGroup();
    if (!is_set) {
      group = var_group;
      is_set = true;
    }
    if (group != var_group)
      return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_synchronizeMulti(VariableCollection vars)
{
  IParallelMng* pm = m_parallel_mng;
  ITimeStats* ts = pm->timeStats();
  Timer::Phase tphase(ts, TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " MultiSync variable";
  if (m_trace_sync) {
    info() << " MultiSynchronize"
           << " stack=" << platform::getStackTrace();
  }
  // Debut de la synchro
  auto& global_on_synchronized = m_variable_synchronizer_mng->onSynchronized();
  bool has_observers = global_on_synchronized.hasObservers() || m_on_synchronized.hasObservers();
  if (has_observers) {
    VariableSynchronizerEventArgs args(vars, this);
    global_on_synchronized.notify(args);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(), "SyncTimer", Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    m_default_message->m_multi_dispatcher->synchronize(vars);
    for (VariableCollection::Enumerator ivar(vars); ++ivar;) {
      (*ivar)->setIsSynchronized();
    }
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("MultiSynchronize", elapsed_time, 1);
  // Fin de la synchro
  if (has_observers) {
    VariableSynchronizerEventArgs args(vars, this, elapsed_time);
    global_on_synchronized.notify(args);
    m_on_synchronized.notify(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
communicatingRanks()
{
  return m_communicating_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
sharedItems(Int32 index)
{
  return m_sync_info->sendInfo().localIds(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
ghostItems(Int32 index)
{
  return m_sync_info->receiveInfo().localIds(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
