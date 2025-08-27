// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFinder2.h                                            (C) 2000-2019 */
/*                                                                           */
/* Classe pour trouver un service donné.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEFINDER2_H
#define ARCANE_SERVICEFINDER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"

#include "arcane/core/IServiceInfo.h"
#include "arcane/core/IFactoryService.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IServiceFactory.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/ServiceInstance.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe utilitaire pour retrouver un ou plusieurs services
 * implémentant l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class ServiceFinderBase2T
{
 public:
  typedef IServiceFactory2T<InterfaceType> FactoryType;
 public:
  ServiceFinderBase2T(IApplication* app,const ServiceBuildInfoBase& sbi)
  : m_application(app), m_service_build_info_base(sbi)
  {
  }

  virtual ~ServiceFinderBase2T(){}

 public:
  /*!
   * \brief Créé une instance du service \a name.
   *
   * Retourne nul si aucun service de ce nom n'existe.
   *
   * \deprecated Utiliser createReference() à la place.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  virtual InterfaceType* create(const String& name)
  {
    return _create(name,m_service_build_info_base);
  }

  /*!
   * \brief Créé une instance du service \a name.
   *
   * Retourne une référence nulle si aucun service de ce nom n'existe.
   */
  virtual Ref<InterfaceType> createReference(const String& name)
  {
    return _createReference(name,m_service_build_info_base);
  }

  /*!
   * \brief Créé une instance du service \a name pour le maillage \a mesh.
   *
   * Cela n'est valide que pour les services de sous-domaine. Pour les autres,
   * cela est sans effet.
   * L'appelant doit détruire ces services.
   * Retourne nul si aucun service de ce nom n'existe.
   *
   * \deprecated Utiliser createReference() à la place.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  virtual InterfaceType* create(const String& name,IMesh* mesh)
  {
    ISubDomain* sd = m_service_build_info_base.subDomain();
    if (!sd)
      return {};
    if (mesh)
      return _create(name,ServiceBuildInfoBase(sd,mesh));
    return _create(name,ServiceBuildInfoBase(sd));
  }

  /*!
   * \brief Créé une instance du service \a name pour le maillage \a mesh.
   *
   * Cela n'est valide que pour les services de sous-domaine. Pour les autres,
   * cela est sans effet.
   * L'appelant doit détruire ces services.
   * Retourne nul si aucun service de ce nom n'existe.
   */
  virtual Ref<InterfaceType> createReference(const String& name,IMesh* mesh)
  {
    ISubDomain* sd = m_service_build_info_base.subDomain();
    if (!sd)
      return {};
    if (mesh)
      return _createReference(name,ServiceBuildInfoBase(sd,mesh));
    return _createReference(name,ServiceBuildInfoBase(sd));
  }

  /*!
   * \brief Instance singleton du service ayant pour interface \a InterfaceType.
   *
   * Retourne nul si aucun service n'est trouvé
   */
  virtual InterfaceType* getSingleton()
  {
    IServiceMng* sm = m_service_build_info_base.serviceParent()->serviceMng();
    SingletonServiceInstanceCollection singleton_services = sm->singletonServices();
    for( typename SingletonServiceInstanceCollection::Enumerator i(singleton_services); ++i; ){
      ISingletonServiceInstance* ssi = (*i).get();
      if (ssi){
        for( typename ServiceInstanceCollection::Enumerator k(ssi->interfaceInstances()); ++k; ){
          IServiceInstance* sub_isi = (*k).get();
          auto m = dynamic_cast< IServiceInstanceT<InterfaceType>* >(sub_isi);
          if (m)
            return m->instance().get();
        }
      }
    }
    return nullptr;
  }

  /*!
   * \brief Créé une instance de chaque service qui implémente \a InterfaceType.
   *
   * L'appelant doit détruire ces services via l'appel à 'operator delete'.
   *
   * \deprecated Utilise ls surcharge prenant en argument un tableau de références.
   */
  ARCCORE_DEPRECATED_2019("Use createAll(Array<ServiceRef<InterfaceType>>&) instead")
  virtual void createAll(Array<InterfaceType*>& instances)
  {
    _createAll(instances,m_service_build_info_base);
  }

  /*!
   * \brief Créé une instance de chaque service qui implémente \a InterfaceType.
   */
  virtual UniqueArray<Ref<InterfaceType>> createAll()
  {
    return _createAll(m_service_build_info_base);
  }

 public:

  SharedArray<FactoryType*> factories()
  {
    SharedArray<FactoryType*> m_factories;
    for( typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j; ){
      IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast< IServiceFactory2T<InterfaceType>* >(sf2);
      //m_application->traceMng()->info() << " FOUND sf2=" << sf2 << " M=" << m;
      if (m){
        m_factories.add(m);
      }
    }
    return m_factories.constView();
  }

  void getServicesNames(Array<String>& names) const
  {
    for( typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j; ){
      IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* true_factory = dynamic_cast< IServiceFactory2T<InterfaceType>* >(sf2);
      if (true_factory){
        IServiceInfo* si = sf2->serviceInfo();
        names.add(si->localName());
        }
    }
  }

 protected:

  InterfaceType* _create(const String& name,const ServiceBuildInfoBase& sbib)
  {
    return _createReference(name,sbib)._release();
  }

  Ref<InterfaceType> _createReference(const String& name,const ServiceBuildInfoBase& sbib)
  {
    for( typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j; ){
      Internal::IServiceFactory2* sf2 = *j;
      IServiceInfo* s = sf2->serviceInfo();
      if (s->localName()!=name)
        continue;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast< IServiceFactory2T<InterfaceType>* >(sf2);
      //m_application->traceMng()->info() << " FOUND sf2=" << sf2 << " M=" << m;
      if (m){
        Ref<InterfaceType> tt = m->createServiceReference(sbib);
        if (!tt.isNull())
          return tt;
      }
    }
    return {};
  }

  void _createAll(Array<InterfaceType*>& instances,const ServiceBuildInfoBase& sbib)
  {
    UniqueArray<Ref<InterfaceType>> ref_instances = _createAll(sbib);
    for( auto& x : ref_instances )
      instances.add(x._release());
  }

  UniqueArray<Ref<InterfaceType>> _createAll(const ServiceBuildInfoBase& sbib)
  {
    UniqueArray<Ref<InterfaceType>> instances;
    for( typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j; ){
      Internal::IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast< IServiceFactory2T<InterfaceType>* >(sf2);
      if (m){
        Ref<InterfaceType> tt = m->createServiceReference(sbib);
        if (tt.get()){
          instances.add(tt);
        }
      }
    }
    return instances;
  }

 protected:

  IApplication* m_application;
  ServiceBuildInfoBase m_service_build_info_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour retrouver un ou plusieurs services
 * implémentant l'interface \a InterfaceType.
 * \deprecated Cette classe ne doit plus être utilisée directement.
 * Il faut utiliser à la place ServiceBuilder.
 */
template<typename InterfaceType,typename ParentType>
class ServiceFinder2T
: public Internal::ServiceFinderBase2T<InterfaceType>
{
 public:
  ServiceFinder2T(IApplication* app,ParentType* parent)
  : Internal::ServiceFinderBase2T<InterfaceType>(app,ServiceBuildInfoBase(parent))
  {
  }

  ~ServiceFinder2T(){}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
