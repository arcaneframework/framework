// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerEventArgs.cc                            (C) 2000-2017 */
/*                                                                           */
/* Arguments des évènements générés par IVariableSynchronizer.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/VariableSynchronizerEventArgs.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs,Real elapsed_time, State state)
: m_unique_variable(var)
, m_var_syncer(vs)
, m_elapsed_time(elapsed_time)
, m_state(state)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs,Real elapsed_time, State state)
: m_unique_variable(nullptr)
, m_var_syncer(vs)
, m_elapsed_time(elapsed_time)
, m_state(state)
{
  m_variables.reserve(vars.count());
  for( VariableCollectionEnumerator v(vars); ++v; )
    m_variables.add(*v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs)
: m_unique_variable(var)
, m_var_syncer(vs)
, m_elapsed_time(0.)
, m_state(State::BeginSynchronize)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs)
: m_unique_variable(nullptr)
, m_var_syncer(vs)
, m_elapsed_time(0.)
, m_state(State::BeginSynchronize)
{
  m_variables.reserve(vars.count());
  for( VariableCollectionEnumerator v(vars); ++v; )
    m_variables.add(*v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
~VariableSynchronizerEventArgs()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste des variables synchronisées.
ConstArrayView<IVariable*> VariableSynchronizerEventArgs::
variables() const
{
  if (m_unique_variable)
    return ConstArrayView<IVariable*>(1,&m_unique_variable);
  return m_variables.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
