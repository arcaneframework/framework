﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerEventArgs.cc                            (C) 2000-2023 */
/*                                                                           */
/* Arguments des évènements générés par IVariableSynchronizer.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableSynchronizerEventArgs.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs,
                              Real elapsed_time, State state)
: m_var_syncer(vs)
{
  initialize(var);
  m_state = state;
  m_elapsed_time = elapsed_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs,
                              Real elapsed_time, State state)
: m_var_syncer(vs)
{
  initialize(vars);
  m_state = state;
  m_elapsed_time = elapsed_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs)
: m_var_syncer(vs)
{
  initialize(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs)
: m_var_syncer(vs)
{
  initialize(vars);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste des variables synchronisées.
ConstArrayView<IVariable*> VariableSynchronizerEventArgs::
variables() const
{
  return m_variables.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerEventArgs::
initialize(const VariableCollection& vars)
{
  _reset();
  m_variables.reserve(vars.count());
  m_compare_status_list.reserve(vars.count());
  for( VariableCollectionEnumerator v(vars); ++v; ){
    m_variables.add(*v);
    m_compare_status_list.add(CompareStatus::Unknown);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerEventArgs::
initialize(IVariable* var)
{
  _reset();
  m_variables.add(var);
  m_compare_status_list.add(CompareStatus::Unknown);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerEventArgs::
_reset()
{
  m_elapsed_time = 0.0;
  m_state = State::BeginSynchronize;
  m_variables.clear();
  m_compare_status_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
