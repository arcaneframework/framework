// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationBuildInfo.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Informations pour construire une instance de IApplication.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ApplicationBuildInfo.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/ApplicationInfo.h"

#include "arcane/core/CaseDatasetSource.h"

#include "arccore/common/internal/FieldProperty.h"
#include "arccore/common/internal/ArccoreApplicationBuildInfoImpl.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void _clamp(Int32& x,Int32 min_value,Int32 max_value)
{
  x = std::min(std::max(x,min_value),max_value);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationBuildInfo::Impl
{
  template <typename T> using FieldProperty = PropertyImpl::FieldProperty<T>;

 public:

  Impl()
  : m_nb_shared_memory_sub_domain(0)
  , m_nb_replication_sub_domain(0)
  , m_nb_processus_sub_domain(0)
  , m_config_file_name("")
  {
    // Fixe une limite en dur pour éviter d'avoir trop de sous-domaines
    // en mémoire partagé (le maximum est en général le nombre de coeurs par
    // noeud)
    m_nb_shared_memory_sub_domain.setValidator([](Int32& x){ _clamp(x,0,1024); });
    m_nb_replication_sub_domain.setValidator([](Int32& x){ x = std::max(x,0); });
    m_nb_processus_sub_domain.setValidator([](Int32& x){ x = std::max(x,0); });
  }

 public:


 public:

  FieldProperty<String> m_message_passing_service;
  FieldProperty<Int32> m_nb_shared_memory_sub_domain;
  FieldProperty<Int32> m_nb_replication_sub_domain;
  FieldProperty<Int32> m_nb_processus_sub_domain;
  FieldProperty<String> m_config_file_name;
  FieldProperty<Int32> m_output_level;
  FieldProperty<Int32> m_verbosity_level;
  FieldProperty<Int32> m_minimal_verbosity_level;
  FieldProperty<bool> m_is_master_has_output_file;
  FieldProperty<String> m_output_directory;
  FieldProperty<String> m_thread_binding_strategy;
  ApplicationInfo m_app_info;
  CaseDatasetSource m_case_dataset_source;
  String m_default_message_passing_service;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo::
ApplicationBuildInfo()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo::
ApplicationBuildInfo(const ApplicationBuildInfo& rhs)
: ArccoreApplicationBuildInfo(rhs)
, m_p(new Impl(*rhs.m_p))
{
}

ApplicationBuildInfo& ApplicationBuildInfo::
operator=(const ApplicationBuildInfo& rhs)
{
  ArccoreApplicationBuildInfo::operator=(rhs);
  if (&rhs != this) {
    delete m_p;
    m_p = new Impl(*(rhs.m_p));
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo::
~ApplicationBuildInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setDefaultValues()
{
  ArccoreApplicationBuildInfo::setDefaultValues();
  {
    String str = m_core->getValue({ "ARCANE_PARALLEL_SERVICE" }, "MessagePassingService", String());
    if (!str.null()) {
      String service_name = str + "ParallelSuperMng";
      PropertyImpl::checkSet(m_p->m_message_passing_service, service_name);
    }
  }
  {
    String str = m_core->getValue({ "ARCANE_NB_THREAD" }, "S", String());
    PropertyImpl::checkSet(m_p->m_nb_shared_memory_sub_domain, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_NB_REPLICATION" }, "R", String());
    PropertyImpl::checkSet(m_p->m_nb_replication_sub_domain, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_NB_SUB_DOMAIN" }, "P", String());
    PropertyImpl::checkSet(m_p->m_nb_processus_sub_domain, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_OUTPUT_LEVEL" }, "OutputLevel",
                                  String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_output_level, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_VERBOSITY_LEVEL", "ARCANE_VERBOSE_LEVEL" }, "VerbosityLevel",
                                  String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_verbosity_level, str);
  }
  {
    String str = m_core->getValue({}, "MinimalVerbosityLevel",
                                  String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_minimal_verbosity_level, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_MASTER_HAS_OUTPUT_FILE" }, "MasterHasOutputFile", "0");
    PropertyImpl::checkSet(m_p->m_is_master_has_output_file, str);
  }
  {
    String str = m_core->getValue({ "ARCANE_OUTPUT_DIRECTORY" }, "OutputDirectory",
                                  String());
    PropertyImpl::checkSet(m_p->m_output_directory, str);
  }
  {
    String str = m_core->getValue({}, "CaseDatasetFileName",
                                  String());
    if (!str.null())
      m_p->m_case_dataset_source.setFileName(str);
  }
  {
    String str = m_core->getValue({ "ARCANE_THREAD_BINDING_STRATEGY" }, "ThreadBindingStrategy",
                                  String());
    PropertyImpl::checkSet(m_p->m_thread_binding_strategy, str);
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
    list1.add(thread_str+"ThreadImplementationService");
    list1.add("TBBThreadImplementationService");
    PropertyImpl::checkSet(m_core->m_thread_implementation_services, list1);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setDefaultServices()
{
  ArccoreApplicationBuildInfo::setDefaultServices();
  bool has_shm = nbSharedMemorySubDomain()>0;
  {
    String def_name = (has_shm) ? "Thread" : "Sequential";
    String default_service_name = def_name+"ParallelSuperMng";
    // Positionne la valeur par défaut si ce n'est pas déjà fait.
    if (m_p->m_default_message_passing_service.null())
      m_p->m_default_message_passing_service = default_service_name;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setMessagePassingService(const String& name)
{
  m_p->m_message_passing_service = name;
}

String ApplicationBuildInfo::
messagePassingService() const
{
  return m_p->m_message_passing_service;
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

Int32 ApplicationBuildInfo::
nbSharedMemorySubDomain() const
{
  return m_p->m_nb_shared_memory_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setNbSharedMemorySubDomain(Int32 v)
{
  m_p->m_nb_shared_memory_sub_domain = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
nbReplicationSubDomain() const
{
  return m_p->m_nb_replication_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setNbReplicationSubDomain(Int32 v)
{
  m_p->m_nb_replication_sub_domain = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
nbProcessusSubDomain() const
{
  return m_p->m_nb_processus_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setNbProcessusSubDomain(Int32 v)
{
  m_p->m_nb_processus_sub_domain = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ApplicationBuildInfo::
configFileName() const
{
  return m_p->m_config_file_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setConfigFileName(const String& v)
{
  m_p->m_config_file_name = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
outputLevel() const
{
  return m_p->m_output_level;
}

void ApplicationBuildInfo::
setOutputLevel(Int32 v)
{
  m_p->m_output_level = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
verbosityLevel() const
{
  return m_p->m_verbosity_level;
}

void ApplicationBuildInfo::
setVerbosityLevel(Int32 v)
{
  m_p->m_verbosity_level = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
minimalVerbosityLevel() const
{
  return m_p->m_minimal_verbosity_level;
}

void ApplicationBuildInfo::
setMinimalVerbosityLevel(Int32 v)
{
  m_p->m_minimal_verbosity_level = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ApplicationBuildInfo::
isMasterHasOutputFile() const
{
  return m_p->m_is_master_has_output_file;
}

void ApplicationBuildInfo::
setIsMasterHasOutputFile(bool v)
{
  m_p->m_is_master_has_output_file = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ApplicationBuildInfo::
outputDirectory() const
{
  return m_p->m_output_directory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setOutputDirectory(const String& v)
{
  m_p->m_output_directory = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ApplicationBuildInfo::
threadBindingStrategy() const
{
  return m_p->m_thread_binding_strategy;
}

void ApplicationBuildInfo::
threadBindingStrategy(const String& v)
{
  m_p->m_thread_binding_strategy = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArccoreApplicationBuildInfo::
addParameter(const String& name,const String& value)
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
  command_line_args.fillParameters(names,values);
  for( Integer i=0, n=names.count(); i<n; ++i ){
    addParameter(names[i],values[i]);
  }
  setDefaultValues();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo& ApplicationBuildInfo::
_internalApplicationInfo()
{
  return m_p->m_app_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ApplicationInfo& ApplicationBuildInfo::
_internalApplicationInfo() const
{
  return m_p->m_app_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setApplicationName(const String& v)
{
  m_p->m_app_info.setApplicationName(v);
}
String ApplicationBuildInfo::
applicationName() const
{
  return m_p->m_app_info.applicationName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setCodeVersion(const VersionInfo& version_info)
{
  m_p->m_app_info.setCodeVersion(version_info);
}

VersionInfo ApplicationBuildInfo::
codeVersion() const
{
  return m_p->m_app_info.codeVersion();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setCodeName(const String& code_name)
{
  m_p->m_app_info.setCodeName(code_name);
}

String ApplicationBuildInfo::
codeName() const
{
  return m_p->m_app_info.codeName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDatasetSource& ApplicationBuildInfo::
caseDatasetSource()
{
  return m_p->m_case_dataset_source;
}

const CaseDatasetSource& ApplicationBuildInfo::
caseDatasetSource() const
{
  return m_p->m_case_dataset_source;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
addDynamicLibrary(const String& lib_name)
{
  m_p->m_app_info.addDynamicLibrary(lib_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
internalSetDefaultMessagePassingService(const String& name)
{
  m_p->m_default_message_passing_service = name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ApplicationBuildInfo::
internalDefaultMessagePassingService() const
{
  return m_p->m_default_message_passing_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

