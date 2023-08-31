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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
VariableSynchronizer(IParallelMng* pm,const ItemGroup& group,
                     Ref<IDataSynchronizeImplementationFactory> implementation_factory)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_item_group(group)
{
  m_sync_info = DataSynchronizeInfo::create();
  if (!implementation_factory.get())
    implementation_factory = arcaneCreateSimpleVariableSynchronizerFactory(pm);
  m_implementation_factory = implementation_factory;

  GroupIndexTable* table = nullptr;
  if (!group.isAllItems())
    table = group.localIdToIndex().get();
  DataSynchronizeDispatcherBuildInfo bi(pm,table,implementation_factory, m_sync_info);
  m_dispatcher = IDataSynchronizeDispatcher::create(bi);
  if (!m_dispatcher)
    ARCANE_FATAL("No synchronizer created");
  m_multi_dispatcher = IDataSynchronizeMultiDispatcher::create(bi);
  if (!m_multi_dispatcher)
    ARCANE_FATAL("No multi synchronizer created");

  {
    String s = platform::getEnvironmentVariable("ARCANE_ALLOW_MULTISYNC");
    if (s=="0" || s=="FALSE" || s=="false")
      m_allow_multi_sync = false;
  }
  {
    String s = platform::getEnvironmentVariable("ARCANE_TRACE_SYNCHRONIZE");
    if (s=="1" || s=="TRUE" || s=="true")
      m_trace_sync = true;
  }
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_AUTO_COMPARE_SYNCHRONIZE", true))
      m_is_compare_sync = (v.value()!=0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
~VariableSynchronizer()
{
  delete m_sync_timer;
  delete m_multi_dispatcher;
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

  m_dispatcher->compute();
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
  Timer::Phase tphase(ts,TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " Sync variable " << var->fullName();
  if (m_trace_sync){
    info() << " Synchronize variable " << var->fullName()
           << " stack=" << platform::getStackTrace();
  }
  // Debut de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(var,this);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(),"SyncTimer",Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    _synchronize(var);
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("Synchronize",elapsed_time,1);
  // Fin de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(var,this,elapsed_time);
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
  if (use_multi && _canSynchronizeMulti(vars)){
    _synchronizeMulti(vars);
  }
  else{
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      synchronize(*ivar);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeResult VariableSynchronizer::
_synchronize(INumericDataInternal* data)
{
  m_dispatcher->beginSynchronize(data, m_is_compare_sync);
  return m_dispatcher->endSynchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_synchronize(IVariable* var)
{
  ARCANE_CHECK_POINTER(var);
  INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data",var->name());
  DataSynchronizeResult result = _synchronize(numapi);
  eDataSynchronizeCompareStatus s = result.compareStatus();
  if (m_is_compare_sync){
    if (s==eDataSynchronizeCompareStatus::Different)
      info() << "Different values name=" << var->name();
    else if (s==eDataSynchronizeCompareStatus::Same)
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
  _synchronize(numapi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  info(4) << "** VariableSynchronizer::changeLocalIds() group=" << m_item_group.name();
  m_sync_info->changeLocalIds(old_to_new_ids);
  m_dispatcher->compute();
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
  if (vars.count()==1)
    return false;
  ItemGroup group;
  bool is_set = false;
  for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
    IVariable* var = *ivar;
    if (var->isPartial())
      return false;
    ItemGroup var_group = var->itemGroup();
    if (!is_set){
      group = var_group;
      is_set = true;
    }
    if (group!=var_group)
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
  Timer::Phase tphase(ts,TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " MultiSync variable";
  if (m_trace_sync){
    info() << " MultiSynchronize"
           << " stack=" << platform::getStackTrace();
  }
  // Debut de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(vars,this);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(),"SyncTimer",Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    m_multi_dispatcher->synchronize(vars);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->setIsSynchronized();
    }
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("MultiSynchronize",elapsed_time,1);
  // Fin de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(vars,this,elapsed_time);
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
