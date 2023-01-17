// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonVariables.cc                                          (C) 2000-2022 */
/*                                                                           */
/* Variables communes décrivant un cas.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CommonVariables.h"
#include "arcane/IModule.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonVariables::
CommonVariables(IModule* c)
: CommonVariables(c->subDomain()->variableMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonVariables::
CommonVariables(ISubDomain* sd)
: CommonVariables(sd->variableMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonVariables::
CommonVariables(IVariableMng* variable_mng)
: m_global_iteration(VariableBuildInfo(variable_mng,"GlobalIteration"))
, m_global_time(VariableBuildInfo(variable_mng,"GlobalTime"))
, m_global_deltat(VariableBuildInfo(variable_mng,"GlobalDeltaT"))
, m_global_old_time(VariableBuildInfo(variable_mng,"GlobalOldTime"))
, m_global_old_deltat(VariableBuildInfo(variable_mng,"GlobalOldDeltaT"))
, m_global_final_time(VariableBuildInfo(variable_mng,"GlobalFinalTime"))
, m_global_old_cpu_time(VariableBuildInfo(variable_mng,"GlobalOldCPUTime"))
, m_global_cpu_time(VariableBuildInfo(variable_mng,"GlobalCPUTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
, m_global_old_elapsed_time(VariableBuildInfo(variable_mng,"GlobalOldElapsedTime"))
, m_global_elapsed_time(VariableBuildInfo(variable_mng,"GlobalElapsedTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CommonVariables::
globalIteration() const
{
  return m_global_iteration();
}

Real CommonVariables::
globalTime() const
{
  return m_global_time();
}

Real CommonVariables::
globalOldTime() const
{
  return m_global_old_time();
}

Real CommonVariables::
globalFinalTime() const
{
  return m_global_final_time();
}

Real CommonVariables::
globalDeltaT() const
{
  return m_global_deltat();
}

Real CommonVariables::
globalOldCPUTime() const
{
  return m_global_old_cpu_time();
}

Real CommonVariables::
globalCPUTime() const
{
  return m_global_cpu_time();
}

Real CommonVariables::
globalOldElapsedTime() const
{
  return m_global_old_elapsed_time();
}

Real CommonVariables::
globalElapsedTime() const
{
  return m_global_elapsed_time();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

