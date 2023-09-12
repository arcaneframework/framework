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
#include "arcane/utils/IMemoryRessourceMng.h"

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
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/accelerator/core/Runner.h"

#include "arcane/impl/DataSynchronizeInfo.h"
#include "arcane/impl/internal/VariableSynchronizerComputeList.h"
#include "arcane/impl/internal/IBufferCopier.h"
#include "arcane/impl/internal/DataSynchronizeMemory.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'une synchronisation.
 *
 * Il est possible d'utiliser plusieurs fois cette instance. Il suffit
 * d'appeler initialize() pour réinitialiser l'instance.
 */
class VariableSynchronizer::SyncMessage
{
 public:

  SyncMessage(const DataSynchronizeDispatcherBuildInfo& bi, VariableSynchronizer* var_syncer)
  : m_variable_synchronizer(var_syncer)
  , m_variable_synchronizer_mng(var_syncer->synchronizeMng())
  , m_dispatcher(IDataSynchronizeDispatcher::create(bi))
  , m_multi_dispatcher(IDataSynchronizeMultiDispatcher::create(bi))
  , m_event_args(var_syncer)
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

  void initialize(IVariable* var)
  {
    _reset();
    m_event_args.initialize(var);
    _addVariable(var);
  }

  void initialize(const VariableCollection& vars)
  {
    _reset();
    m_event_args.initialize(vars);
    for (VariableCollectionEnumerator v(vars); ++v;)
      _addVariable(*v);
  }

  Int32 nbVariable() const { return m_variables.size(); }
  ConstArrayView<IVariable*> variables() const { return m_variables; }

  //! Effectue la synchronisation
  void synchronize()
  {
    Int32 nb_var = m_variables.size();
    if (nb_var == 0)
      return;
    if (nb_var == 1) {
      bool is_compare_sync = m_variable_synchronizer_mng->isCompareSynchronize();
      m_synchronize_result = synchronizeData(m_data_list[0], is_compare_sync);
    }
    if (nb_var >= 2) {
      m_multi_dispatcher->synchronize(m_variables);
    }
    for (IVariable* var : m_variables)
      var->setIsSynchronized();
  }

  DataSynchronizeResult synchronizeData(INumericDataInternal* data, bool is_compare_sync)
  {
    m_dispatcher->beginSynchronize(data, is_compare_sync);
    return m_dispatcher->endSynchronize();
  }
  const DataSynchronizeResult& result() const { return m_synchronize_result; }
  VariableSynchronizerEventArgs& eventArgs() { return m_event_args; }

 private:

  IVariableSynchronizer* m_variable_synchronizer = nullptr;
  IVariableSynchronizerMng* m_variable_synchronizer_mng = nullptr;
  Ref<IDataSynchronizeDispatcher> m_dispatcher;
  IDataSynchronizeMultiDispatcher* m_multi_dispatcher = nullptr;
  VariableSynchronizerEventArgs m_event_args;
  UniqueArray<IVariable*> m_variables;
  UniqueArray<INumericDataInternal*> m_data_list;
  DataSynchronizeResult m_synchronize_result;

 private:

  void _reset()
  {
    m_variables.clear();
    m_data_list.clear();
  }

  void _addVariable(IVariable* var)
  {
    INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
    if (!numapi)
      ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data", var->name());
    m_variables.add(var);
    m_data_list.add(numapi);
  }
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

  Ref<IBufferCopier> buffer_copier;
  if (table)
    buffer_copier = makeRef<IBufferCopier>(new TableBufferCopier(table));
  else
    buffer_copier = makeRef<IBufferCopier>(new DirectBufferCopier());

  auto* internal_pm = m_parallel_mng->_internalApi();
  Runner* runner = internal_pm->defaultRunner();
  bool is_accelerator_aware = internal_pm->isAcceleratorAware();
  IMemoryAllocator* allocator = nullptr;
  // Si le IParallelMng gère la mémoire des accélérateurs alors on alloue le
  // buffer sur le device. On pourrait utiliser la mémoire managée mais certaines
  // implémentations MPI (i.e: BXI) ne le supportent pas.
  if (runner && is_accelerator_aware) {
    m_runner = runner;
    buffer_copier->setRunQueue(internal_pm->defaultQueue());
    allocator = platform::getDataMemoryRessourceMng()->getAllocator(eMemoryRessource::Device);
  }

  DataSynchronizeMemory* memory = new DataSynchronizeMemory(allocator);
  Ref<DataSynchronizeMemory> ref_memory = makeRef<DataSynchronizeMemory>(memory);

  // Créé une instance de l'implémentation
  Ref<IDataSynchronizeImplementation> sync_impl = m_implementation_factory->createInstance();
  sync_impl->setDataSynchronizeInfo(m_sync_info.get());

  DataSynchronizeDispatcherBuildInfo bi(m_parallel_mng, sync_impl, m_sync_info, ref_memory, buffer_copier);
  return new SyncMessage(bi, this);
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

  _setCurrentDevice();
  m_default_message->compute();
  if (m_is_verbose)
    info() << "End compute dispatcher Date=" << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_doSynchronize(SyncMessage* message)
{
  IParallelMng* pm = m_parallel_mng;
  ITimeStats* ts = pm->timeStats();
  Timer::Phase tphase(ts, TP_Communication);

  _setCurrentDevice();

  // Envoi l'évènement de début de la synchro
  VariableSynchronizerEventArgs& event_args = message->eventArgs();
  _sendBeginEvent(event_args);

  {
    Timer::Sentry ts2(m_sync_timer);
    message->synchronize();
  }

  Int32 nb_var = message->nbVariable();
  // Si une seule variable, affiche le résutat de la comparaison de
  // la synchronisation
  if (nb_var == 1 && m_variable_synchronizer_mng->isCompareSynchronize()) {
    IVariable* var = message->variables()[0];
    eDataSynchronizeCompareStatus s = message->result().compareStatus();
    if (s == eDataSynchronizeCompareStatus::Different) {
      event_args.setCompareStatus(0, VariableSynchronizerEventArgs::CompareStatus::Different);
      info() << "Different values name=" << var->name();
    }
    else if (s == eDataSynchronizeCompareStatus::Same) {
      event_args.setCompareStatus(0, VariableSynchronizerEventArgs::CompareStatus::Same);
      info() << "Same values name=" << var->name();
    }
    else
      info() << "Unknown values name=" << var->name();
  }

  // Fin de la synchro
  _sendEndEvent(event_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronize(IVariable* var)
{
  m_default_message->initialize(var);

  IParallelMng* pm = m_parallel_mng;
  debug(Trace::High) << " Proc " << pm->commRank() << " Sync variable " << var->fullName();
  if (m_trace_sync) {
    info() << " Synchronize variable " << var->fullName()
           << " stack=" << platform::getStackTrace();
  }
  _doSynchronize(m_default_message);
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
  return m_default_message->synchronizeData(data, is_compare_sync);
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
_synchronizeMulti(const VariableCollection& vars)
{
  m_default_message->initialize(vars);

  IParallelMng* pm = m_parallel_mng;
  debug(Trace::High) << " Proc " << pm->commRank() << " MultiSync variable";
  if (m_trace_sync) {
    info() << " MultiSynchronize"
           << " stack=" << platform::getStackTrace();
  }

  _doSynchronize(m_default_message);
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

void VariableSynchronizer::
_sendBeginEvent(VariableSynchronizerEventArgs& args)
{
  _checkCreateTimer();
  args.setState(VariableSynchronizerEventArgs::State::BeginSynchronize);
  _sendEvent(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_sendEndEvent(VariableSynchronizerEventArgs& args)
{
  ARCANE_CHECK_POINTER(m_sync_timer);
  Real elapsed_time = m_sync_timer->lastActivationTime();
  m_parallel_mng->stat()->add("Synchronize", elapsed_time, 1);
  args.setState(VariableSynchronizerEventArgs::State::EndSynchronize);
  args.setElapsedTime(elapsed_time);
  _sendEvent(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_sendEvent(VariableSynchronizerEventArgs& args)
{
  m_variable_synchronizer_mng->onSynchronized().notify(args);
  m_on_synchronized.notify(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_checkCreateTimer()
{
  if (!m_sync_timer)
    m_sync_timer = new Timer(m_parallel_mng->timerMng(), "SyncTimer", Timer::TimerReal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le device associé à notre RunQueue comme le device courant.
 *
 * Si on utilise une RunQueue, positionne le device associé à celui
 * de cette RunQueue. Cela permet de garantir que les allocations mémoires
 * effectuées lors des synchronisations seront sur le bon device.
 */
void VariableSynchronizer::
_setCurrentDevice()
{
  if (m_runner)
    m_runner->setAsCurrentDevice();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
