// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Application.h                                               (C) 2000-2026 */
/*                                                                           */
/* IApplication Implementation.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_APPLICATION_H
#define ARCANE_IMPL_APPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/NullThreadMng.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/XmlNode.h"

#include "arccore/base/ReferenceCounter.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceAndModuleFactoryMng;
class IProcessorAffinityService;
class ISymbolizerService;
class ApplicationBuildInfo;
class ConcurrencyApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Supervisor.
 */
class ARCANE_IMPL_EXPORT Application
: public IApplication
{
  class CoreApplication;

 public:

  explicit Application(IArcaneMain*);
  ~Application() override;

 public:

  void build() override;
  void initialize() override;

 public:
  
  IBase* objectParent() const override { return nullptr; }
  String objectNamespaceURI() const override { return m_namespace_uri; }
  String objectLocalName() const override { return m_local_name; }
  VersionInfo objectVersion() const override { return VersionInfo(1,0,0); }

 public:
	
  ITraceMng* traceMng() const override { return m_trace.get(); }
  IRessourceMng* ressourceMng() const override { return m_ressource_mng; }
  IServiceMng* serviceMng() const override { return m_service_mng; }

  IParallelSuperMng* parallelSuperMng() override { return m_parallel_super_mng.get(); }
  IParallelSuperMng* sequentialParallelSuperMng() override
  {
    return m_sequential_parallel_super_mng;
  }
  IIOMng* ioMng() override { return m_io_mng; }
  IConfigurationMng* configurationMng() const override { return m_configuration_mng; }
  IDataFactory* dataFactory() override;
  IDataFactoryMng* dataFactoryMng() const override;
  const ApplicationInfo& applicationInfo() const override { return m_exe_info; }
  const ApplicationBuildInfo& applicationBuildInfo() const override;
  const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const override;
  const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const override;
  String versionStr() const override { return m_version_str; }
  String majorAndMinorVersionStr() const override { return m_major_and_minor_version_str; }
  String mainVersionStr() const override { return m_main_version_str; }
  String targetinfoStr() const override { return m_targetinfo_str; }
  String applicationName() const override { return m_application_name; }
  String codeName() const override { return m_code_name; }
  String userName() const override { return m_user_name; }
  String userConfigPath() const override { return m_user_config_path; }
  IMainFactory* mainFactory() const override { return m_main_factory; }

  ByteConstSpan configBuffer() const override { return asBytes(m_config_bytes.constSpan()); }
  ByteConstSpan userConfigBuffer() const override { return asBytes(m_user_config_bytes.constSpan()); }
  
  SessionCollection sessions() override { return m_sessions; }
  void addSession(ISession* s) override;
  void removeSession(ISession* s) override;

  ServiceFactory2Collection serviceFactories2() override;
  ModuleFactoryInfoCollection moduleFactoryInfos() override;

  Ref<ICodeService> getCodeService(const String& file_name) override;
  bool hasGarbageCollector() const override { return m_has_garbage_collector; }

  IPhysicalUnitSystemService* getPhysicalUnitSystemService() override
  {
    return m_physical_unit_system_service.get();
  }

  ITraceMngPolicy* getTraceMngPolicy() override { return m_trace_policy; }
  ITraceMng* createAndInitializeTraceMng(ITraceMng* parent_trace,
                                         const String& file_suffix) override;

 private:
  ApplicationInfo m_exe_info; //!< Executable information
  String m_namespace_uri;
  String m_local_name;
  NullThreadImplementation m_null_thread_implementation;
  IArcaneMain* m_arcane_main = nullptr;
  IMainFactory* m_main_factory = nullptr; //!< Main factory
  IServiceMng* m_service_mng = nullptr; //!< Service manager
  Ref<IParallelSuperMng> m_parallel_super_mng; //!< Parallelism manager
  IParallelSuperMng* m_sequential_parallel_super_mng = nullptr; //!< Sequential parallelism manager.
  ReferenceCounter<ITraceMng> m_trace; //!< Trace manager
  IRessourceMng* m_ressource_mng = nullptr; //!< Resource manager
  IIOMng* m_io_mng = nullptr; //!< Input/output manager
  IConfigurationMng* m_configuration_mng = nullptr;
  Ref<IDataFactoryMng> m_data_factory_mng; //!< Data factory
  String m_version_str; //!< Configuration info
  String m_main_version_str; //!< Version in Major.minor.beta format
  String m_major_and_minor_version_str; //!< Version M.m
  String m_targetinfo_str; //!< Configuration info
  String m_code_name;
  String m_application_name; //!< Application name
  String m_user_name; //!< User name
  String m_user_config_path; //!< User configuration directory
  SessionList m_sessions; //!< List of sessions
  ServiceFactoryInfoCollection m_main_service_factory_infos; //!< Array of service factories
  ModuleFactoryInfoCollection m_main_module_factory_infos; //!< Array of module factories
  bool m_has_garbage_collector = false;
  ITraceMngPolicy* m_trace_policy = nullptr;

 private:

  bool m_is_init = false; //!< \e true if already initialized
  UniqueArray<Byte> m_config_bytes; //!< File containing the configuration
  UniqueArray<Byte> m_user_config_bytes; //!< File containing the user configuration
  ScopedPtrT<IXmlDocumentHolder> m_config_document; //!< Configuration DOM tree
  ScopedPtrT<IXmlDocumentHolder> m_user_config_document; //!< User configuration DOM tree
  XmlNode m_config_root_element; //!< Configuration root element
  XmlNode m_user_config_root_element; //!< User configuration root element
  //bool m_is_info_disabled;
  bool m_is_master = false;
  Ref<IPhysicalUnitSystemService> m_physical_unit_system_service;
  Ref<IOnlineDebuggerService> m_online_debugger;
  Ref<IProfilingService> m_profiling_service;

  IServiceAndModuleFactoryMng* m_service_and_module_factory_mng = nullptr;

  Ref<IProcessorAffinityService> m_processor_affinity_service;
  Ref<IPerformanceCounterService> m_performance_counter_service;
  Ref<IParallelSuperMng> m_owned_sequential_parallel_super_mng;
  std::unique_ptr<ConcurrencyApplication> m_core_application;

 private:

  void _openUserConfig();
  void _initDataInitialisationPolicy();
  template<typename InterfaceType> Ref<InterfaceType>
  _tryCreateService(const StringList& names,String* found_name);
  void _readCodeConfigurationFile();
  void _setCoreServices();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
