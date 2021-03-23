// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceAndModuleFactoryMng.cc                               (C) 2000-2019 */
/*                                                                           */
/* Gestionnaire des fabriques de services et modules.                        */
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
 * \brief Gestionnaire des fabriques de services et modules.
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
    // Il ne faut pas détruire les instances de IServiceFactory2 car elle
    // sont gérées par le IServiceInfo correspondant.

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

  //! Liste des informations sur les des services
  List<IServiceInfo*> m_service_infos;
  //! Liste des informations sur les fabriques des services
  List<IServiceFactoryInfo*> m_service_factory_infos;
  //! Liste des informations sur les fabriques des modules
  List<IModuleFactoryInfo*> m_module_factory_infos;
  //! Liste des informations sur les fabriques des services (V2)
  List<Internal::IServiceFactory2*> m_service_factories2;

  //! Liste des IServiceFactoryInfo à détruire.
  UniqueArray<IServiceFactoryInfo*> m_deletable_service_factory_list;
  //! Liste des IModuleFactoryInfo à détruire.
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
 * \brief Détruit le gestionnaire.
 *
 * Détruit le gestionnaire de message et les gestionnaires de configuration.
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
  // Les fabriques globales ne doivent pas être détruites par nous.
  info() << "Add global service factory name=" << sfi->serviceInfo()->localName();
  m_p->addServiceFactory(sfi,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceAndModuleFactoryMng::
addGlobalFactory(IModuleFactoryInfo* mfi)
{
  // Les fabriques globales ne doivent pas être détruites par nous.
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

  // Enregistre toutes les fabriques utilisant ServiceRegisterer

  while(sr){
    // Detecte les problèmes de boucle infinie (eg: si deux services ont le même nom)
    // Désormais les contrôles dans ServiceRegisterer devrait toutefois suffire
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
 * \brief Ajoute la fabrique spécifiée par \a sr.
 *
 * La fabrique peut être celle d'un service ou d'un module. Dans le
 * premier cas, elle est ajoutée à \a m_service_factory_info. Dans le
 * second cas, elle est ajoutée à \a m_module_factory_info.
 */
void ServiceAndModuleFactoryMng::
_addFactoryFromServiceRegisterer(ServiceRegisterer* sr)
{
  ARCANE_CHECK_POINTER2(sr,"ServiceRegisterer");

  bool is_ok = false;

  // Regarde si \a sr n'a pas déjà été traité.
  // Cela peut arriver avec le chargement dynamique si createAllServiceRegistererFactories()
  // est appelé plusieurs fois.
  if (m_service_registerer_done_set.find(sr)!=m_service_registerer_done_set.end())
    return;
  m_service_registerer_done_set.insert(sr);

  // Tente de créer le IServiceInfo suivant les différentes méthodes possibles.
  // Si sr->moduleFactoryWithPropertyFunction() est non nul, il s'agit d'une fabrique de module.
  // Sinon, il s'agit obligatoirement d'un service.
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

    // Indique qu'il faudra détruire l'instance \a si
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
