// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs,Real elapsed_time, State state)
: m_var_syncer(vs)
, m_elapsed_time(elapsed_time)
, m_state(state)
{
  setVariable(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs,Real elapsed_time, State state)
: m_var_syncer(vs)
, m_elapsed_time(elapsed_time)
, m_state(state)
{
  setVariables(vars);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs)
: m_var_syncer(vs)
{
  setVariable(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerEventArgs::
VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs)
: m_var_syncer(vs)
{
  setVariables(vars);
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
  return m_variables.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerEventArgs::
setVariables(const VariableCollection& vars)
{
  m_variables.clear();
  m_variables.reserve(vars.count());
  for( VariableCollectionEnumerator v(vars); ++v; )
    m_variables.add(*v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerEventArgs::
setVariable(IVariable* var)
{
  m_variables.clear();
  m_variables.add(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
