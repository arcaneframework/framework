// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultBackwardMng.cc                                       (C) 2000-2016 */
/*                                                                           */
/* Implémentation par défaut d'une stratégie de retour-arrière.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/DefaultBackwardMng.h"

#include "arcane/IVariableFilter.h"
#include "arcane/IVariable.h"
#include "arcane/IVariableMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/CommonVariables.h"
#include "arcane/Timer.h"
#include "arcane/VariableCollection.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/impl/MemoryDataReaderWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RestoreVariableFilter
: public IVariableFilter
{
public:

  virtual bool applyFilter(IVariable& var)
  {
    if (!var.isUsed())
      return false;
    if (var.property() & (IVariable::PTemporary|IVariable::PNoRestore))
      return false;
    return true;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultBackwardMng::
DefaultBackwardMng(ITraceMng* trace,ISubDomain* sub_domain)
: m_trace(trace)
, m_sub_domain(sub_domain)
, m_filter(0)
, m_data_io(0)
, m_backward_time(-1)
, m_period(0)
, m_first_save(true)
, m_action_refused(true)
, m_sequence(SEQNothing)
{
  ARCANE_ASSERT((m_trace),("ITraceMng pointer null"));
  ARCANE_ASSERT((m_sub_domain),("ISubDomain pointer null"));

  m_filter = new RestoreVariableFilter();

  m_data_io = new MemoryDataReaderWriter(trace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultBackwardMng::
~DefaultBackwardMng()
{
  delete m_filter;

  delete m_data_io;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
clear()
{
  m_data_io->free();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
goBackward()
{
  if (!m_action_refused){
    m_trace->pfatal() << "Trying to go backward during an action phase";
  }

  // Si déjà en train de faire un retour arrière, on ne peut pas continuer
  if (m_sequence == SEQRestore) {
    m_trace->pfatal() << "Impossible to go backward while it is already going backward";
  }

  m_sequence = SEQRestore;

  m_trace->info() << "Execution of the backward mode !";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
_checkValidAction()
{
  if(m_action_refused)
    throw FatalErrorException(A_FUNCINFO,"Action requested outside the authorized period");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DefaultBackwardMng::
checkAndApplyRestore()
{
  _checkValidAction();
  if (m_sequence==SEQRestore){
    _restore();
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DefaultBackwardMng::
checkAndApplySave(bool is_forced)
{
  _checkValidAction();
  m_action_refused = true;
  _checkSave(is_forced);
  if (m_sequence==SEQSave){
    _save();
    return true;
  }

  if (m_sequence==SEQForceSave){
    m_trace->info() << "Physical time of the last backward reached or passed. Save is forced";
    _save();
    return true;
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
_restore()
{
  m_backward_time = m_sub_domain->commonVariables().globalTime();

  m_sub_domain->variableMng()->readVariables(m_data_io,m_filter);

  m_sequence = SEQLock;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
_save()
{
  Timer::Action ts_action(m_sub_domain,"RestoreSave");

  m_trace->info() << "Saving variable before going backward";

  m_sub_domain->variableMng()->writeVariables(m_data_io,m_filter);

  m_sequence = SEQNothing;

  m_backward_time = -1.;

  const Real     var_mem = m_sub_domain->variableMng()->exportSize(VariableList());
  const Real max_var_mem = m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax,var_mem);

  m_trace->info() << "Memory allocated for the variables (in Mo): used="
                  << var_mem << " max=" << max_var_mem;

  m_first_save = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
beginAction()
{
  if(!m_action_refused)
  {
    m_trace->pfatal() << "Begin of action requested before the end";
  }

  m_action_refused = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
_checkSave(bool is_forced)
{
  if (m_period == 0) {
    m_sequence = SEQNothing;
    return;
  }

  if (m_sequence == SEQRestore)
    return;

  const Real time = m_sub_domain->commonVariables().globalTime();

  if (m_backward_time >= 0. && m_backward_time < time) {
    m_sequence = SEQForceSave;
    return;
  }

  // Ne sauve pas les infos tant que le retour-arrière est actif
  if (m_backward_time >= 0.) {
    m_sequence = SEQLock;
    return;
  }

  const Integer iteration = m_sub_domain->commonVariables().globalIteration();

  bool do_save = false;

  if (iteration > 1){
    do_save |= m_first_save;
    do_save |= (iteration % m_period) == 0;
  }

  if (is_forced)
    do_save = true;

  if(do_save) {
    m_sequence = SEQSave;
  }
  else {
    m_sequence = SEQNothing;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultBackwardMng::
endAction()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

