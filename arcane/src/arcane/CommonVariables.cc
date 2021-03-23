// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonVariables.cc                                          (C) 2000-2006 */
/*                                                                           */
/* Variables communes décrivant un cas.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/CommonVariables.h"
#include "arcane/IModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonVariables::
CommonVariables(IModule* c)
: m_global_iteration(VariableBuildInfo(c->subDomain(),"GlobalIteration"))
, m_global_time(VariableBuildInfo(c->subDomain(),"GlobalTime"))
, m_global_deltat(VariableBuildInfo(c->subDomain(),"GlobalDeltaT"))
, m_global_old_time(VariableBuildInfo(c->subDomain(),"GlobalOldTime"))
, m_global_old_deltat(VariableBuildInfo(c->subDomain(),"GlobalOldDeltaT"))
, m_global_final_time(VariableBuildInfo(c->subDomain(),"GlobalFinalTime"))
, m_global_old_cpu_time(VariableBuildInfo(c->subDomain(),"GlobalOldCPUTime"))
, m_global_cpu_time(VariableBuildInfo(c->subDomain(),"GlobalCPUTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
, m_global_old_elapsed_time(VariableBuildInfo(c->subDomain(),"GlobalOldElapsedTime"))
, m_global_elapsed_time(VariableBuildInfo(c->subDomain(),"GlobalElapsedTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonVariables::
CommonVariables(ISubDomain* sd)
: m_global_iteration(VariableBuildInfo(sd,"GlobalIteration"))
, m_global_time(VariableBuildInfo(sd,"GlobalTime"))
, m_global_deltat(VariableBuildInfo(sd,"GlobalDeltaT"))
, m_global_old_time(VariableBuildInfo(sd,"GlobalOldTime"))
, m_global_old_deltat(VariableBuildInfo(sd,"GlobalOldDeltaT"))
, m_global_final_time(VariableBuildInfo(sd,"GlobalFinalTime"))
, m_global_old_cpu_time(VariableBuildInfo(sd,"GlobalOldCPUTime"))
, m_global_cpu_time(VariableBuildInfo(sd,"GlobalCPUTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
, m_global_old_elapsed_time(VariableBuildInfo(sd,"GlobalOldElapsedTime"))
, m_global_elapsed_time(VariableBuildInfo(sd,"GlobalElapsedTime",IVariable::PNoRestore|IVariable::PExecutionDepend))
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

