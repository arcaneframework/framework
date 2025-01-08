// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Application.h                                               (C) 2000-2025 */
/*                                                                           */
/* Implémentation IApplication.                                              */
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

#include "arcane/IApplication.h"
#include "arcane/XmlNode.h"

#include "arccore/base/ReferenceCounter.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Superviseur.
 */
class ARCANE_IMPL_EXPORT Application
: public IApplication
{
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
  ApplicationInfo m_exe_info; //!< Informations sur l'exécutable
  String m_namespace_uri;
  String m_local_name;
  NullThreadImplementation m_null_thread_implementation;
  IArcaneMain* m_arcane_main = nullptr;
  IMainFactory* m_main_factory = nullptr; //!< Manufacture principale
  IServiceMng* m_service_mng = nullptr; //!< Gestionnaire des services
  Ref<IParallelSuperMng> m_parallel_super_mng; //!< Gestionnaire du parallélisme
  IParallelSuperMng* m_sequential_parallel_super_mng = nullptr; //!< Gestionnaire du parallélisme séquentiel.
  ReferenceCounter<ITraceMng> m_trace; //!< Gestionnaire de traces
  IRessourceMng* m_ressource_mng = nullptr; //!< Gestionnaire de ressources
  IIOMng* m_io_mng = nullptr; //!< Gestionnaire des entrées/sorties
  IConfigurationMng* m_configuration_mng = nullptr;
  Ref<IDataFactoryMng> m_data_factory_mng; //!< Fabrique des données
  String m_version_str; //!< Infos sur la configuration
  String m_main_version_str; //!< Version sous la forme Majeur.mineur.beta
  String m_major_and_minor_version_str; //!< Version M.m
  String m_targetinfo_str; //!< Infos sur la configuration
  String m_code_name;
  String m_application_name; //!< Nom de l'application
  String m_user_name; //!< Nom de l'utilisateur
  String m_user_config_path; //!< Répertoire de configuration utilisateur
  SessionList m_sessions; //!< Liste des sessions
  ServiceFactoryInfoCollection m_main_service_factory_infos; //!< Tableau des fabriques de service
  ModuleFactoryInfoCollection m_main_module_factory_infos; //!< Tableau des fabriques de module
  bool m_has_garbage_collector;
  ITraceMngPolicy* m_trace_policy = nullptr;

 private:

  bool m_is_init; //!< \e true si déjà initialisé
  UniqueArray<Byte> m_config_bytes; //!< Fichier contenant la configuration
  UniqueArray<Byte> m_user_config_bytes; //!< Fichier contenant la configuration utilisateur
  ScopedPtrT<IXmlDocumentHolder> m_config_document; //!< Arbre DOM de la configuration
  ScopedPtrT<IXmlDocumentHolder> m_user_config_document; //!< Arbre DOM de la configuration utilisateur
  XmlNode m_config_root_element; //!< Elément racine de la configuration
  XmlNode m_user_config_root_element; //!< Elément racine de la configuration utilisateur
  //bool m_is_info_disabled;
  bool m_is_master;
  Ref<IPhysicalUnitSystemService> m_physical_unit_system_service;
  //! Nom du service utilisé pour gérer les threads
  String m_used_thread_service_name;
  //! Nom du service utilisé pour gérer les tâches
  String m_used_task_service_name;
  Ref<IOnlineDebuggerService> m_online_debugger;
  Ref<IProfilingService> m_profiling_service;

  IServiceAndModuleFactoryMng* m_service_and_module_factory_mng;

  Ref<IProcessorAffinityService> m_processor_affinity_service;
  Ref<IStackTraceService> m_stack_trace_service;
  Ref<ISymbolizerService> m_symbolizer_service;
  Ref<IThreadImplementationService> m_thread_implementation_service;
  Ref<IThreadImplementation> m_thread_implementation;
  Ref<IPerformanceCounterService> m_performance_counter_service;
  Ref<IParallelSuperMng> m_owned_sequential_parallel_super_mng;
  Ref<ITaskImplementation> m_task_implementation;

 private:

  void _openUserConfig();
  void _initDataInitialisationPolicy();
  template<typename InterfaceType> Ref<InterfaceType>
  _tryCreateService(const StringList& names,String* found_name);
  template<typename InterfaceType> Ref<InterfaceType>
  _tryCreateServiceUsingInjector(const StringList& names,String* found_name);
  UniqueArray<String> _stringListToArray(const StringList& slist) const;
  void _readCodeConfigurationFile();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

