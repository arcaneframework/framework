// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceAndModuleFactoryMng.cc                               (C) 2000-2019 */
/*                                                                           */
/* Manager of service and module factories.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/List.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Array.h"

#include "arcane/ServiceUtils.h"
#include "arcane/ServiceInfo.h"
#include "arcane/ServiceRegisterer.h"
#include "arcane/IServiceFactory.h"
#include "arcane/IModuleFactory.h"
#include "arcane/ModuleProperty.h"
#include "arcane/IServiceAndModuleFactoryMng.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manager of service and module factories.
 */
class ARCANE_IMPL_EXPORT ServiceAndModuleFactoryMng
: public TraceAccessor
, public IServiceAndModuleFactoryMng
{
 public:

  class Impl;

 public:

  ServiceAndModuleFactoryMng(ITraceMng* tm);
  virtual ~ServiceAndModuleFactoryMng();

 public:
  
  virtual void createAllServiceRegistererFactories();

  virtual ServiceFactoryInfoCollection serviceFactoryInfos() const;
  virtual ServiceFactory2Collection serviceFactories2() const;
  virtual ModuleFactoryInfoCollection moduleFactoryInfos() const;

  virtual void addGlobalFactory(IServiceFactoryInfo* sfi);
  virtual void addGlobalFactory(IModuleFactoryInfo* mfi);

 private:

  std::set<ServiceRegisterer*> m_service_registerer_done_set;
  Impl* m_p;

 private:

  void _addFactoryFromServiceRegisterer(ServiceRegisterer* sr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceAndModuleFactoryMng::Impl
{
 public:

  ServiceFactoryInfoCollection serviceFactoryInfos() const { return m_service_factory_infos; }
  ServiceFactory2Collection serviceFactories2() const { return m_service_factories2; }
  ModuleFactoryInfoCollection moduleFactoryInfos() const { return m_module_factory_infos; }

 public:

  ~Impl()
  {
    // The instances of IServiceFactory2 must not be destroyed because they
    // are managed by the corresponding IServiceInfo.

    for( List<IServiceInfo*>::Enumerator i(m_service_infos); ++i; )
      delete *i;

    for( IServiceFactoryInfo* sfi : m_deletable_service_factory_list )
      delete sfi;
  }

  void addServiceFactory(IServiceFactoryInfo* sfi,bool need_delete)
  {
    if (m_service_factory_set.find(sfi)!=m_service_factory_set.end()){
      std::cout << "Service Factory is already referenced\n";
      return;
    }
    m_service_factory_set.insert(sfi);
    m_service_factory_infos.add(sfi);
    if (need_delete)
      m_deletable_service_factory_list.add(sfi);
    IServiceInfo* si = sfi->serviceInfo();
    for( ServiceFactory2Collection::Enumerator j(si->factories()); ++j; ){
      Internal::IServiceFactory2* sf2 = *j;
      m_service_factories2.add(sf2);
    }
  }

  void addModuleFactory(IModuleFactoryInfo* mfi)
  {
    if (m_module_factory_set.find(mfi)!=m_module_factory_set.end()){
      std::cout << "Module Factory is already referenced\n";
      return;
    }
    m_module_factory_set.insert(mfi);
    m_module_factory_infos.add(mfi);
    m_deletable_module_factory_list.add(ModuleFactoryReference(mfi));
  }

  void registerServiceInfoForDelete(IServiceInfo* si)
  {
    m_service_infos.add(si);
  }

 private:

  //! List of service information.
  List<IServiceInfo*> m_service_infos;
  //! List of service factory information.
  List<IServiceFactoryInfo*> m_service_factory_infos;
  //! List of module factory information.
  List<IModuleFactoryInfo*> m_module_factory_infos;
  //! List of service factory information (V2).
  List<Internal::IServiceFactory2*> m_service_factories2;

  //! List of IServiceFactoryInfo to be destroyed.
  UniqueArray<IServiceFactoryInfo*> m_deletable_service_factory_list;
  //! List of IModuleFactoryInfo to be destroyed.
  UniqueArray<ModuleFactoryReference> m_deletable_module_factory_list;

  std::set<IServiceFactoryInfo*> m_service_factory_set;
  std::set<IModuleFactoryInfo*> m_module_factory_set;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IServiceAndModuleFactoryMng*
arcaneCreateServiceAndModuleFactoryMng(ITraceMng* tm)
{
  IServiceAndModuleFactoryMng* sm = new ServiceAndModuleFactoryMng(tm);
  return sm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceAndModuleFactoryMng::
ServiceAndModuleFactoryMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Destroys the manager.
 *
 * Destroys the message manager and configuration managers.
 */
ServiceAndModuleFactoryMng::
~ServiceAndModuleFactoryMng()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceFactoryInfoCollection ServiceAndModuleFactoryMng::
serviceFactoryInfos() const
{
  return m_p->serviceFactoryInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceFactory2Collection ServiceAndModuleFactoryMng::
serviceFactories2() const
{
  return m_p->serviceFactories2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactoryInfoCollection ServiceAndModuleFactoryMng::
moduleFactoryInfos() const
{
 return m_p->moduleFactoryInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceAndModuleFactoryMng::
addGlobalFactory(IServiceFactoryInfo* sfi)
{
  // Global factories must not be destroyed by us.
  info() << "Add global service factory name=" << sfi->serviceInfo()->localName();
  m_p->addServiceFactory(sfi,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceAndModuleFactoryMng::
addGlobalFactory(IModuleFactoryInfo* mfi)
{
  // Global factories must not be destroyed by us.
  info() << "Add global module factory name=" << mfi->moduleName();
  m_p->addModuleFactory(mfi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceAndModuleFactoryMng::
createAllServiceRegistererFactories()
{
  ServiceRegisterer* sr = ServiceRegisterer::firstService();
  if (!sr)
    log() << "WARNING: No registered service";

  OStringStream oss;
  std::set<ServiceRegisterer*> registered_services;

  // Registers all factories using ServiceRegisterer

  while(sr){
    // Detects infinite loop problems (e.g., if two services have the same name)
    // Controls in ServiceRegisterer should now be sufficient.
    if (registered_services.find(sr) == registered_services.end()) {
      oss() << "\t" << sr->name() << '\n';
      registered_services.insert(sr);
    }
    else {
      cout << "=== Registered service factories ===\n"
           << " Registered service count: " << registered_services.size() << " / " << ServiceRegisterer::nbService()
           << "====================================\n"
           << oss.str() 
           << "====================================" << endl;
      ARCANE_FATAL("Infinite loop in service registration");
    }

    _addFactoryFromServiceRegisterer(sr);

    sr = sr->nextService();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds the factory specified by \a sr.
 *
 * The factory can be for a service or a module. In the
 * first case, it is added to \a m_service_factory_info. In the
 * second case, it is added to \a m_module_factory_info.
 */
void ServiceAndModuleFactoryMng::
_addFactoryFromServiceRegisterer(ServiceRegisterer* sr)
{
  ARCANE_CHECK_POINTER2(sr,"ServiceRegisterer");

  bool is_ok = false;

  // Checks if \a sr has already been processed.
  // This can happen with dynamic loading if createAllServiceRegistererFactories()
  // is called multiple times.
  if (m_service_registerer_done_set.find(sr)!=m_service_registerer_done_set.end())
    return;
  m_service_registerer_done_set.insert(sr);

  // Tries to create the IServiceInfo using the different possible methods.
  // If sr->moduleFactoryWithPropertyFunction() is not null, it is a module factory.
  // Otherwise, it must be a service.
  if (sr->moduleFactoryWithPropertyFunction()){
    IModuleFactoryInfo* mfi = (*sr->moduleFactoryWithPropertyFunction())(sr->moduleProperty());
    if (mfi){
      m_p->addModuleFactory(mfi);
      //trace->info() << "Add module factory for '" << si->localName() << "' mfi=" << mfi;
      is_ok = true;
    }
  }
  else{
    auto property_info_func = sr->infoCreatorWithPropertyFunction();
    if (!property_info_func)
      ARCANE_FATAL("Null PropertyFunc for ServiceRegisterer");

    IServiceInfo* si = (*property_info_func)(sr->serviceProperty());
    if (!si)
      ARCANE_FATAL("Null ServiceInfo created by ServiceRegisterer");

    // Indicates that the instance \a si must be destroyed.
    m_p->registerServiceInfoForDelete(si);

    IServiceFactoryInfo* sfi = si->factoryInfo();

    if (sfi){
      m_p->addServiceFactory(sfi,true);
      //trace->info() << "Add service factory for '" << si->localName() << "' sfi=" << sfi;
      is_ok = true;
    }
  }

  if (!is_ok){
    info() << "WARNING: ServiceRegisterer does not have a valid create function name=" << sr->name();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
