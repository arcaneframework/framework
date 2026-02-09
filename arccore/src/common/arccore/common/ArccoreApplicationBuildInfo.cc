// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreApplicationBuildInfo.cc                              (C) 2000-2026 */
/*                                                                           */
/* Informations pour construire une instance de IApplication.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/ArccoreApplicationBuildInfo.h"
#include "arccore/common/CommandLineArguments.h"
#include "arccore/common/internal/FieldProperty.h"
#include "arccore/common/internal/ArccoreApplicationBuildInfoImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void _clamp(Int32& x, Int32 min_value, Int32 max_value)
  {
    x = std::min(std::max(x, min_value), max_value);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArccoreApplicationBuildInfoImpl::ArccoreApplicationBuildInfoImpl()
: m_nb_task_thread(-1)
{
  // Fixe une limite pour le nombre de tâches
  m_nb_task_thread.setValidator([](Int32& x) { _clamp(x, -1, 512); });
}

String ArccoreApplicationBuildInfoImpl::
getValue(const UniqueArray<String>& env_values, const String& param_name,
         const String& default_value)
{
  return m_property_key_values.getValue(env_values, param_name, default_value);
}

void ArccoreApplicationBuildInfoImpl::
addKeyValue(const String& name, const String& value)
{
  m_property_key_values.add(name, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArccoreApplicationBuildInfo::
ArccoreApplicationBuildInfo()
: m_core(new ArccoreApplicationBuildInfoImpl())
{
}

ArccoreApplicationBuildInfo::
ArccoreApplicationBuildInfo(const ArccoreApplicationBuildInfo& rhs)
: m_core(new ArccoreApplicationBuildInfoImpl(*rhs.m_core))
{
}

ArccoreApplicationBuildInfo& ArccoreApplicationBuildInfo::
operator=(const ArccoreApplicationBuildInfo& rhs)
{
  if (&rhs != this) {
    delete m_core;
    m_core = new ArccoreApplicationBuildInfoImpl(*(rhs.m_core));
  }
  return (*this);
}

ArccoreApplicationBuildInfo::
~ArccoreApplicationBuildInfo()
{
  delete m_core;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
setDefaultValues()
{
  {
    String str = m_core->getValue({ "ARCANE_NB_TASK" }, "T", String());
    PropertyImpl::checkSet(m_core->m_nb_task_thread, str);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
setDefaultServices()
{
  {
    String str = m_core->getValue({ "ARCANE_TASK_IMPLEMENTATION" }, "TaskService", "TBB");
    String service_name = str + "TaskImplementation";
    PropertyImpl::checkSet(m_core->m_task_implementation_services, service_name);
  }
  {
    StringList list1;
    String thread_str = m_core->getValue({ "ARCANE_THREAD_IMPLEMENTATION" }, "ThreadService", "Std");
    list1.add(thread_str + "ThreadImplementationService");
    list1.add("TBBThreadImplementationService");
    PropertyImpl::checkSet(m_core->m_thread_implementation_services, list1);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
setTaskImplementationService(const String& name)
{
  StringList s;
  s.add(name);
  m_core->m_task_implementation_services = s;
}
void ArccoreApplicationBuildInfo::
setTaskImplementationServices(const StringList& names)
{
  m_core->m_task_implementation_services = names;
}
StringList ArccoreApplicationBuildInfo::
taskImplementationServices() const
{
  return m_core->m_task_implementation_services;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
setThreadImplementationService(const String& name)
{
  StringList s;
  s.add(name);
  m_core->m_thread_implementation_services = s;
}
void ArccoreApplicationBuildInfo::
setThreadImplementationServices(const StringList& names)
{
  m_core->m_thread_implementation_services = names;
}
StringList ArccoreApplicationBuildInfo::
threadImplementationServices() const
{
  return m_core->m_thread_implementation_services;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ArccoreApplicationBuildInfo::
nbTaskThread() const
{
  return m_core->m_nb_task_thread;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
setNbTaskThread(Int32 v)
{
  m_core->m_nb_task_thread = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
addParameter(const String& name, const String& value)
{
  m_core->addKeyValue(name, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
parseArgumentsAndSetDefaultsValues(const CommandLineArguments& command_line_args)
{
  // On ne récupère que les arguments du style:
  //   -A,x=b,y=c
  StringList names;
  StringList values;
  command_line_args.fillParameters(names, values);
  for (Integer i = 0, n = names.count(); i < n; ++i) {
    addParameter(names[i], values[i]);
  }
  setDefaultValues();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
