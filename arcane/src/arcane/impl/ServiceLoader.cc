// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceLoader.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Service loader for available services in the code.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CriticalSection.h"

#include "arcane/IServiceLoader.h"
#include "arcane/IServiceMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/Service.h"
#include "arcane/IModuleFactory.h"
#include "arcane/IModule.h"

#include "arcane/ServiceInfo.h"
#include "arcane/IService.h"

#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service loader in the architecture.
 */
class ServiceLoader
: public IServiceLoader
{
 public:

 public:

  ServiceLoader();

  ~ServiceLoader() override;

  //! Loads available application services
  void loadApplicationServices(IApplication*) override;
  //! Loads available session services
  void loadSessionServices(ISession*) override;
  //! Loads available subdomain services in the subdomain \a sd
  void loadSubDomainServices(ISubDomain*parent) override;
  //! Loads available modules
  void loadModules(ISubDomain* sd,bool all_modules) override;

  void initializeModuleFactories(ISubDomain* sd) override;

  bool loadSingletonService(ISubDomain* sd,const String& name) override;

 private:
  
  void _loadServices(IApplication* application,const ServiceBuildInfoBase& sbib);
  SingletonServiceInstanceRef
  _createSingletonInstance(IServiceMng* sm,IServiceInfo* si,const ServiceBuildInfoBase& sbi);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceLoader::
ServiceLoader()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceLoader::
~ServiceLoader()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IServiceLoader*
arcaneCreateServiceLoader()
{
  IServiceLoader* icl = new ServiceLoader();
  return icl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceLoader::
loadApplicationServices(IApplication* parent)
{
  ITraceMng* trace = parent->traceMng();
  trace->log() << "Loading Application Services";
  _loadServices(parent,ServiceBuildInfoBase(parent));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceLoader::
loadSessionServices(ISession* parent)
{
  ITraceMng* trace = parent->traceMng();
  trace->log() << "Loading Session Services";
  _loadServices(parent->application(),ServiceBuildInfoBase(parent));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceLoader::
loadSubDomainServices(ISubDomain* parent)
{
  ITraceMng* trace = parent->traceMng();
  trace->log() << "Loading SubDomain Services";
  {
    _loadServices(parent->application(),ServiceBuildInfoBase(parent));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SingletonServiceInstanceRef ServiceLoader::
_createSingletonInstance(IServiceMng* sm,IServiceInfo* si,const ServiceBuildInfoBase& sbi)
{
  ITraceMng* tm = sm->traceMng();
  IServiceFactoryInfo* sfi = si->factoryInfo();
  SingletonServiceInstanceRef instance;

  // If the singleton factory exists, we use it. Otherwise, we use
  // the old mechanism. Normally, the singleton factory always exists
  // unless we are using an Arcane version with an old version of Axlstar.
  Internal::ISingletonServiceFactory* ssf = si->singletonFactory();
  if (ssf){
    instance = ssf->createSingletonServiceInstance(sbi);
    if (instance.get()){
      if (!sfi->isSingleton())
        tm->info() << "WARNING: singleton service loading'"
                   << si->localName() << "' which is not specified as singleton.";
    }
  }

  if (instance.get()){
    sm->addSingletonInstance(instance);
    String local_name = si->localName();
    VersionInfo vi = si->version();
    StringCollection implemented_interfaces = si->implementedInterfaces();
    tm->log() << "Loading singleton service " << local_name
              << " (Version " << vi << ")"
              << " (Type " << typeid(instance.get()).name() << ")"
              << " N=" << implemented_interfaces.count();
    for( StringCollection::Enumerator sc(implemented_interfaces); ++sc; ){
      tm->log() << " (Interface implemented '" << *sc << "'";
    }
  }

  return instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ServiceLoader::
loadSingletonService(ISubDomain* sd,const String& name)
{
  // Normally, the service must be a singleton to be loaded
  // in this manner. Nevertheless, for compatibility reasons, we
  // allow loading all services in singleton mode and we
  // display a warning. Eventually, do_all will be false and it will be necessary to specify
  // that the service is a singleton
  const bool do_all = true;
  ITraceMng* trace = sd->traceMng();
  IServiceMng* service_mng = sd->serviceMng();

  // Checks that no instance of the same name exists.
  // If so, we do nothing and display a warning.
  // Maybe a fatal would be more appropriate.
  SingletonServiceInstanceRef old_instance = service_mng->singletonServiceReference(name);
  if (old_instance.get()){
    trace->warning() << "An instance of singleton service; name: '" << name << "' already exists."
                     << " The second instance will not be created !";
    return true;
  }

  ServiceFactory2Collection service_factory_infos(sd->application()->serviceFactories2());
  for( ServiceFactory2Collection::Enumerator i(service_factory_infos); ++i; ){
    Internal::IServiceFactory2* sf2 = *i;
    IServiceInfo* si = sf2->serviceInfo();
    IServiceFactoryInfo* sfi = si->factoryInfo();
    if (!do_all)
      if (!sfi->isSingleton())
        continue;
    if (si->localName()!=name)
      continue;

    ServiceBuildInfoBase sbi(sd);
    auto instance = _createSingletonInstance(service_mng,si,sbi);
    if (instance.get())
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Loads services in the base manager.
 */
void ServiceLoader::
_loadServices(IApplication* application,const ServiceBuildInfoBase& sbib)
{
  // Instantiates singleton services that load automatically
  // (they have the isAutoload() property set to true).
  IServiceMng* service_mng = sbib.serviceParent()->serviceMng();
 
  ServiceFactory2Collection service_factory_infos(application->serviceFactories2());
  for( ServiceFactory2Collection::Enumerator i(service_factory_infos); ++i; ){
    Internal::IServiceFactory2* sf2 = *i;
    IServiceInfo* si = sf2->serviceInfo();
    IServiceFactoryInfo* sfi = si->factoryInfo();
    if (!sfi->isSingleton())
      continue;
    if (!sfi->isAutoload())
      continue;

    _createSingletonInstance(service_mng,si,sbib);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceLoader::
loadModules(ISubDomain* sd,bool all_modules)
{
  CriticalSection cs(sd->threadMng());

  ITraceMng* trace = sd->traceMng();
  IApplication* app = sd->application();
  ModuleFactoryInfoCollection module_factory_infos(app->moduleFactoryInfos());
  for( ModuleFactoryInfoCollection::Enumerator i(module_factory_infos); ++i; ){
    IModuleFactoryInfo* sf = *i;
    bool is_autoload = sf->isAutoload();
    if (sf->isAutoload() || all_modules){
      Ref<IModule> module = sf->createModule(sd,sd->defaultMeshHandle());
      if (module.get())
        trace->info() << "Loading module " << module->name()
                      << " (Version " << module->versionInfo() << ")"
                      << ((is_autoload) ? " (autoload)" : " ");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceLoader::
initializeModuleFactories(ISubDomain* sd)
{
  IApplication* app = sd->application();
  ModuleFactoryInfoCollection module_factory_infos(app->moduleFactoryInfos());
  for( ModuleFactoryInfoCollection::Enumerator i(module_factory_infos); ++i; ){
    IModuleFactoryInfo* sf = *i;
    sf->initializeModuleFactory(sd);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
