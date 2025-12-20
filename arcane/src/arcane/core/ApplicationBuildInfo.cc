// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationBuildInfo.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Informations pour construire une instance de IApplication.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ApplicationBuildInfo.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/ApplicationInfo.h"

#include "arcane/CaseDatasetSource.h"

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

class PropertyImpl
{
 public:

  template <typename DataType>
  class Property
  {
   public:

    explicit Property(DataType default_value)
    : m_value(default_value)
    , m_default_value(default_value)
    {}
    Property()
    : Property(DataType())
    {}
    Property<DataType>& operator=(const DataType& v)
    {
      setValue(v);
      return (*this);
    }
    operator DataType() const { return m_value; }

   public:

    void setValue(const DataType& v)
    {
      if (m_validator) {
        DataType copy(v);
        m_validator(copy);
        m_value = copy;
      }
      else
        m_value = v;
      m_has_value = true;
    }
    DataType value() const { return m_value; }
    bool isValueSet() const { return m_has_value; }
    void setValidator(std::function<void(DataType&)>&& func) { m_validator = func; }

   private:

    DataType m_value;
    DataType m_default_value;
    bool m_has_value = false;
    std::function<void(DataType&)> m_validator;
  };

  class Int32Value
  {
   public:

    explicit Int32Value(Int32 v)
    : value(v)
    {}
    operator Int32() const { return value; }

   public:

    Int32Value minValue(Int32 x)
    {
      return Int32Value(std::max(value, x));
    }
    Int32Value maxValue(Int32 x)
    {
      return Int32Value(std::min(value, x));
    }

   public:

    Int32 value;
  };

  static Int32Value getInt32(const String& str_value, Int32 default_value)
  {
    Int32 v = default_value;
    if (!str_value.null()) {
      bool is_bad = builtInGetValue(v, str_value);
      if (is_bad)
        v = default_value;
    }
    return Int32Value(v);
  }
  static void checkSet(Property<bool>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    bool v = 0;
    bool is_bad = builtInGetValue(v, str_value);
    if (!is_bad)
      p.setValue(v);
  }
  static void checkSet(Property<Int32>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    Int32 v = 0;
    bool is_bad = builtInGetValue(v, str_value);
    if (!is_bad)
      p.setValue(v);
  }
  static void checkSet(Property<StringList>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    StringList s;
    s.add(str_value);
    p.setValue(s);
  }
  static void checkSet(Property<StringList>& p, const StringList& str_values)
  {
    if (p.isValueSet())
      return;
    p.setValue(str_values);
  }
  static void checkSet(Property<String>& p, const String& str_value)
  {
    if (p.isValueSet())
      return;
    if (str_value.null())
      return;
    p.setValue(str_value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationBuildInfo::Impl
{
  template <typename T> using Property = PropertyImpl::Property<T>;

 public:

  class NameValuePair
  {
   public:
    NameValuePair(const String& n,const String& v) : name(n), value(v){}
    String name;
    String value;
  };
 public:
  Impl()
  : m_nb_task_thread(-1), m_nb_shared_memory_sub_domain(0),
    m_nb_replication_sub_domain(0), m_nb_processus_sub_domain(0),
    m_config_file_name("")
  {
    // Fixe une limite pour le nombre de tâches
    m_nb_task_thread.setValidator([](Int32& x){ _clamp(x,-1,512); });
    // Fixe une limite en dur pour éviter d'avoir trop de sous-domaines
    // en mémoire partagé (le maximum est en général le nombre de coeurs par
    // noeud)
    m_nb_shared_memory_sub_domain.setValidator([](Int32& x){ _clamp(x,0,1024); });
    m_nb_replication_sub_domain.setValidator([](Int32& x){ x = std::max(x,0); });
    m_nb_processus_sub_domain.setValidator([](Int32& x){ x = std::max(x,0); });
  }

 public:

  /*!
   * \brief Récupère la valeur d'une option.
   *
   * L'ordre de récupération est le suivant :
   * - si \a param_name est non nul, regarde s'il existe une valeur
   * dans \a m_values associée à ce paramètre. Si oui, on retourne cette
   * valeur.
   * - pour chaque nom \a x de \a env_values, regarde si une variable
   * d'environnement \a x existe et retourne sa valeur si c'est le cas.
   * - si aucune des méthodes précédente n'a fonctionné, retourne
   * la valeur \a default_value.
   */
  String getValue(const UniqueArray<String>& env_values, const String& param_name,
                  const String& default_value)
  {
    if (!param_name.null()) {
      String v = _searchParam(param_name);
      if (!v.null())
        return v;
    }
    for (const auto& x : env_values) {
      String ev = platform::getEnvironmentVariable(x);
      if (!ev.null())
        return ev;
    }
    return default_value;
  }

 public:

  Property<String> m_message_passing_service;
  Property<StringList> m_task_implementation_services;
  Property<StringList> m_thread_implementation_services;
  Property<Int32> m_nb_task_thread;
  Property<Int32> m_nb_shared_memory_sub_domain;
  Property<Int32> m_nb_replication_sub_domain;
  Property<Int32> m_nb_processus_sub_domain;
  Property<String> m_config_file_name;
  Property<Int32> m_output_level;
  Property<Int32> m_verbosity_level;
  Property<Int32> m_minimal_verbosity_level;
  Property<bool> m_is_master_has_output_file;
  Property<String> m_output_directory;
  Property<String> m_thread_binding_strategy;
  UniqueArray<NameValuePair> m_values;
  ApplicationInfo m_app_info;
  CaseDatasetSource m_case_dataset_source;
  String m_default_message_passing_service;

 private:

  String _searchParam(const String& param_name)
  {
    String v;
    // Une option peut être présente plusieurs fois. Prend la dernière.
    for (const auto& x : m_values) {
      if (x.name == param_name)
        v = x.value;
    }
    return v;
  }
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
: m_p(new Impl(*rhs.m_p))
{
}

ApplicationBuildInfo& ApplicationBuildInfo::
operator=(const ApplicationBuildInfo& rhs)
{
  if (&rhs!=this){
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
  {
    String str = m_p->getValue({ "ARCANE_NB_TASK" }, "T", String());
    PropertyImpl::checkSet(m_p->m_nb_task_thread, str);
  }
  {
    String str = m_p->getValue({ "ARCANE_NB_THREAD" }, "S", String());
    PropertyImpl::checkSet(m_p->m_nb_shared_memory_sub_domain, str);
  }
  {
    String str = m_p->getValue({ "ARCANE_NB_REPLICATION" }, "R", String());
    PropertyImpl::checkSet(m_p->m_nb_replication_sub_domain, str);
  }
  {
    String str = m_p->getValue({ "ARCANE_NB_SUB_DOMAIN" }, "P", String());
    PropertyImpl::checkSet(m_p->m_nb_processus_sub_domain, str);
  }
  {
    String str = m_p->getValue( { "ARCANE_OUTPUT_LEVEL" }, "OutputLevel",
                               String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_output_level, str);
  }
  {
    String str = m_p->getValue( { "ARCANE_VERBOSITY_LEVEL", "ARCANE_VERBOSE_LEVEL" }, "VerbosityLevel",
                               String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_verbosity_level, str);
  }
  {
    String str = m_p->getValue( { }, "MinimalVerbosityLevel",
                               String::fromNumber(Trace::UNSPECIFIED_VERBOSITY_LEVEL));
    PropertyImpl::checkSet(m_p->m_minimal_verbosity_level, str);
  }
  {
    String str = m_p->getValue({ "ARCANE_MASTER_HAS_OUTPUT_FILE" }, "MasterHasOutputFile", "0");
    PropertyImpl::checkSet(m_p->m_is_master_has_output_file, str);
  }
  {
    String str = m_p->getValue( { "ARCANE_OUTPUT_DIRECTORY" }, "OutputDirectory",
                               String());
    PropertyImpl::checkSet(m_p->m_output_directory, str);
  }
  {
    String str = m_p->getValue( { }, "CaseDatasetFileName",
                                String() );
    if (!str.null())
      m_p->m_case_dataset_source.setFileName(str);
  }
  {
    String str = m_p->getValue( { "ARCANE_THREAD_BINDING_STRATEGY" }, "ThreadBindingStrategy",
                               String());
    PropertyImpl::checkSet(m_p->m_thread_binding_strategy, str);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setDefaultServices()
{
  bool has_shm = nbSharedMemorySubDomain()>0;
  {
    String str = m_p->getValue( { "ARCANE_TASK_IMPLEMENTATION" }, "TaskService", "TBB");
    String service_name = str + "TaskImplementation";
    PropertyImpl::checkSet(m_p->m_task_implementation_services, service_name);
  }
  {
    StringList list1;
    String thread_str = m_p->getValue( { "ARCANE_THREAD_IMPLEMENTATION" }, "ThreadService" ,"Std");
    list1.add(thread_str+"ThreadImplementationService");
    list1.add("TBBThreadImplementationService");
    PropertyImpl::checkSet(m_p->m_thread_implementation_services, list1);
  }
  {
    String def_name = (has_shm) ? "Thread" : "Sequential";
    String default_service_name = def_name+"ParallelSuperMng";
    // Positionne la valeur par défaut si ce n'est pas déjà fait.
    if (m_p->m_default_message_passing_service.null())
      m_p->m_default_message_passing_service = default_service_name;

    String str = m_p->getValue({ "ARCANE_PARALLEL_SERVICE" }, "MessagePassingService", String());
    if (!str.null()) {
      String service_name = str + "ParallelSuperMng";
      PropertyImpl::checkSet(m_p->m_message_passing_service, service_name);
    }
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

void ApplicationBuildInfo::
setTaskImplementationService(const String& name)
{
  StringList s;
  s.add(name);
  m_p->m_task_implementation_services = s;
}
void ApplicationBuildInfo::
setTaskImplementationServices(const StringList& names)
{
  m_p->m_task_implementation_services = names;
}
StringList ApplicationBuildInfo::
taskImplementationServices() const
{
  return m_p->m_task_implementation_services;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setThreadImplementationService(const String& name)
{
  StringList s;
  s.add(name);
  m_p->m_thread_implementation_services = s;
}
void ApplicationBuildInfo::
setThreadImplementationServices(const StringList& names)
{
  m_p->m_thread_implementation_services = names;
}
StringList ApplicationBuildInfo::
threadImplementationServices() const
{
  return m_p->m_thread_implementation_services;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ApplicationBuildInfo::
nbTaskThread() const
{
  return m_p->m_nb_task_thread;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
setNbTaskThread(Int32 v)
{
  m_p->m_nb_task_thread = v;
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

void ApplicationBuildInfo::
addParameter(const String& name,const String& value)
{
  m_p->m_values.add(Impl::NameValuePair(name,value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationBuildInfo::
parseArguments(const CommandLineArguments& command_line_args)
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

