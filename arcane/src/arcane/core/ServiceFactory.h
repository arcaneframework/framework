// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFactory.h                                            (C) 2000-2025 */
/*                                                                           */
/* Manufacture des services.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERVICEFACTORY_H
#define ARCANE_CORE_SERVICEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ExternalRef.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ISession.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceBuildInfo.h"

#include "arcane/core/IServiceFactory.h"
#include "arcane/core/ServiceRegisterer.h"
#include "arcane/core/ServiceInfo.h"
#include "arcane/core/IService.h"
#include "arcane/core/ServiceProperty.h"
#include "arcane/core/ServiceInstance.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une instance de service en C#.
 */
class DotNetServiceInstance
: public IServiceInstance
{
 public:
  DotNetServiceInstance(IServiceInfo* si)
  : m_service_info(si), m_handle(nullptr){}
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  IServiceInfo* serviceInfo() const override
  {
    return m_service_info;
  }
  void setDotNetHandle(ExternalRef handle) { m_handle = handle; }
  ExternalRef _internalDotNetHandle() const override { return m_handle; }
 private:
  std::atomic<Int32> m_nb_ref = 0;
  IServiceInfo* m_service_info;
  ExternalRef m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Instance d'un service.
 */
template<typename InterfaceType>
class ServiceInstanceT
: public IServiceInstanceT<InterfaceType>
{
 public:
  ServiceInstanceT(Ref<InterfaceType> i,IServiceInfo* si)
  : m_instance(i), m_service_info(si){}
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  Ref<InterfaceType> instance() override
  {
    return m_instance;
  }
  IServiceInfo* serviceInfo() const override
  {
    return m_service_info;
  }
 private:
  std::atomic<Int32> m_nb_ref = 0;
  Ref<InterfaceType> m_instance;
  IServiceInfo* m_service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Infos sur la fabrication d'un service ou d'un module.
 */
class ARCANE_CORE_EXPORT ServiceFactoryInfo
: public IServiceFactoryInfo
{
 public:

  explicit ServiceFactoryInfo(IServiceInfo* si)
  : m_service_info(si), m_is_autoload(false), m_is_singleton(false) {}
  ~ServiceFactoryInfo() override {}

 public:
  
  IServiceInfo* serviceInfo() const override { return m_service_info; }

  bool isAutoload() const override { return m_is_autoload; }
  bool isSingleton() const override { return m_is_singleton; }

  virtual bool isModule() const { return false; }
  virtual void initializeModuleFactory(ISubDomain*) {}
  virtual IModule* createModule(ISubDomain*,IMesh*) { return nullptr; }

 public:

  void setAutoload(bool v) { m_is_autoload = v; }
  void setSingleton(bool v) { m_is_singleton = v; }
  void initProperties(int v)
  {
    if (v & SFP_Singleton)
      setSingleton(v);
    if (v & SFP_Autoload)
      setAutoload(v);
  }
  void initProperties(){}

 private:
  
  IServiceInfo* m_service_info;
  bool m_is_autoload;
  bool m_is_singleton;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un fonctor de création d'une instance de service
 * correspondant à l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class IServiceInterfaceFactory
{
 public:
  virtual ~IServiceInterfaceFactory() = default;
 public:
  //! Créé une instance du service .
  virtual Ref<InterfaceType> createReference(const ServiceBuildInfo& sbi) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique de service.
 *
 * Cette classe implémente IServiceFactory2 et IServiceFactory2T pour
 * l'interface \a InterfaceType.
 */

template<typename InterfaceType>
class ServiceFactory2TV2
: public IServiceFactory2T<InterfaceType>
{
 public:

  ServiceFactory2TV2(IServiceInfo* si,IServiceInterfaceFactory<InterfaceType>* sub_factory)
  : m_service_info(si), m_sub_factory(sub_factory), m_type_flags(si->usageType())
  {
  }

  ~ServiceFactory2TV2() override
  {
    delete m_sub_factory;
  }

  ServiceInstanceRef createServiceInstance(const ServiceBuildInfoBase& sbi) override
  {
    return _create(this->createServiceReference(sbi));
  }

  Ref<InterfaceType> createServiceReference(const ServiceBuildInfoBase& sbi) override
  {
    if (!(m_type_flags & sbi.creationType()))
      return {};
    return _createReference(sbi);
  }

  IServiceInfo* serviceInfo() const override
  {
    return m_service_info;
  }

 protected:

  IServiceInfo* m_service_info;
  IServiceInterfaceFactory<InterfaceType>* m_sub_factory;
  int m_type_flags;

 private:
  
  InterfaceType* _createInstance(const ServiceBuildInfoBase& sbib)
  {
    InterfaceType* it = m_sub_factory->createInstance(ServiceBuildInfo(m_service_info,sbib));
    return it;
  }

  Ref<InterfaceType> _createReference(const ServiceBuildInfoBase& sbib)
  {
    return m_sub_factory->createReference(ServiceBuildInfo(m_service_info,sbib));
  }

  ServiceInstanceRef _create(Ref<InterfaceType> it)
  {
    IServiceInstance* x = (!it) ? nullptr : new ServiceInstanceT<InterfaceType>(it,m_service_info);
    return ServiceInstanceRef::createRef(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour le service \a ServiceType pour l'interface \a InterfaceType.
 */
template<typename ServiceType,typename InterfaceType>
class ServiceInterfaceFactory
: public IServiceInterfaceFactory<InterfaceType>
{
 public:

  Ref<InterfaceType> createReference(const ServiceBuildInfo& sbi) override
  {
    ServiceType* st = new ServiceType(sbi);
    st->build();
    return makeRef<InterfaceType>(st);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCANE_CORE_EXPORT IServiceInstanceAdder
{
 public:
  virtual ~IServiceInstanceAdder() = default;
  virtual void addInstance(ServiceInstanceRef instance) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base des fabriques pour les services singleton.
 *
 * Il faut dériver de cette classe et implémenter _createInstance().
 */
class ARCANE_CORE_EXPORT SingletonServiceFactoryBase
: public ISingletonServiceFactory
{
 public:
  class ServiceInstance;
 public:
  explicit SingletonServiceFactoryBase(IServiceInfo* si) : m_service_info(si){}
 public:

  //! Créé un service singleton
  Ref<ISingletonServiceInstance> createSingletonServiceInstance(const ServiceBuildInfoBase& sbib) override;

  //! Retourne le IServiceInfo associé à cette fabrique.
  IServiceInfo* serviceInfo() const override { return m_service_info; }
 protected:
  virtual ServiceInstanceRef _createInstance(const ServiceBuildInfoBase& sbi,IServiceInstanceAdder* instance_adder) =0;
 private:
  IServiceInfo* m_service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour le service singleton de type \a ServiceType implémentant
 * les interfaces \a Interfaces.
 */
template<typename ServiceType,typename ... Interfaces>
class SingletonServiceFactory
: public SingletonServiceFactoryBase
{
  /*!
   * \brief Classe utilitaire permettant de créér une instance
   * de IServiceInstance pour chaque interface de \a Interfaces.
   */
  class Helper
  {
   public:
    Helper(ServiceType* service,IServiceInfo* si,IServiceInstanceAdder* adder)
    : m_service(service), m_service_info(si), m_adder(adder) {}
   private:
    //! Surcharge pour 1 interface
    template<typename InterfaceType> void _create()
    {
      InterfaceType* x = m_service;
      // ATTENTION: la référence suivante ne doit pas détruire 'm_service' car
      // ce dernier a déjà un ServiceReference qui a été construit lors de
      // l'appel à _createInstance().
      auto x_ref = Ref<InterfaceType>::_createNoDestroy(x);
      auto instance = new ServiceInstanceT<InterfaceType>(x_ref,m_service_info);
      // TODO: indiquer qu'il ne faut pas détruire la référence.
      auto instance_ref = ServiceInstanceRef::createRef(instance);
      m_adder->addInstance(instance_ref);
    }
    //! Surcharge pour 2 interfaces ou plus
    template<typename I1,typename I2,typename ... OtherInterfaces>
    void _create()
    {
      _create<I1>();
      // Applique la récursivité sur les types restants
      _create<I2,OtherInterfaces...>();
    }
   public:
    void createInterfaceInstances()
    {
      _create<Interfaces...>();
    }
   private:
    ServiceType* m_service;
    IServiceInfo* m_service_info;
    IServiceInstanceAdder* m_adder;
  };
 public:
  explicit SingletonServiceFactory(IServiceInfo* si) : SingletonServiceFactoryBase(si){}
 protected:
  ServiceInstanceRef _createInstance(const ServiceBuildInfoBase& sbib,IServiceInstanceAdder* instance_adder) override
  {
    ServiceBuildInfo sbi(serviceInfo(),sbib);
    ServiceType* st = new ServiceType(sbi);
    st->build();
    auto st_ref = makeRef(st);
    Helper ssf(st,serviceInfo(),instance_adder);
    ssf.createInterfaceInstances();
    IServiceInstance* si = new ServiceInstanceT<ServiceType>(st_ref,serviceInfo());
    return ServiceInstanceRef::createRef(si);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant d'enregistrer une fabrique pour un service
 * implémentant l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class ServiceInterfaceRegisterer
{
 public:

  typedef InterfaceType Interface;

  explicit ServiceInterfaceRegisterer(const char* name)
  : m_name(name), m_namespace_name(nullptr)
  {
  }

  ServiceInterfaceRegisterer(const char* namespace_name,const char* name)
  : m_name(name), m_namespace_name(namespace_name)
  {
  }

 public:

  //! Enregistre dans \a si une fabrique pour créer une instance du service \a ServiceType
  template<typename ServiceType> void
  registerToServiceInfo(ServiceInfo* si) const
  {
    IServiceInterfaceFactory<InterfaceType>* factory = new ServiceInterfaceFactory<ServiceType,InterfaceType>();
    if (m_namespace_name)
      si->addImplementedInterface(String(m_namespace_name)+String("::")+String(m_name));
    else
      si->addImplementedInterface(m_name);
    si->addFactory(new ServiceFactory2TV2<InterfaceType>(si,factory));
  }

 private:

  const char* m_name;
  const char* m_namespace_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant de créer et d'enregistrer les fabriques pour un service.
 */
template<typename ServiceType>
class ServiceAllInterfaceRegisterer
{
 private:

  //! Surcharge pour 1 interface
  template<typename InterfaceType> static void
  _create(ServiceInfo* si,const InterfaceType& i1)
  {
    i1.template registerToServiceInfo<ServiceType>(si);
  }
  //! Surcharge pour 2 interfaces ou plus
  template<typename I1,typename I2,typename ... OtherInterfaces>
  static void _create(ServiceInfo* si,const I1& i1,const I2& i2,const OtherInterfaces& ... args)
  {
    _create<I1>(si,i1);
    // Applique la récursivité sur les types restants
    _create<I2,OtherInterfaces...>(si,i2,args...);
  }

 public:

  //! Enregistre dans le service les fabriques pour les interfacs \a Interfaces
  template<typename ... Interfaces> static void
  registerToServiceInfo(ServiceInfo* si, const Interfaces& ... args)
  {
    si->setSingletonFactory(new Internal::SingletonServiceFactory<ServiceType,typename Interfaces::Interface ... >(si));
    _create(si,args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour déclarer une interface lors de l'enregistrement d'un service.
 *
 * Cette macro s'utilise dans la macro ARCANE_REGISTER_SERVICE.
 *
 * L'appel est comme suit:
 *
 \code
 * ARCANE_SERVICE_INTERFACE(ainterface)
 \endcode
 *
 * \a ainterface est le nom de l'interface (sans les guillemets). Il
 * peut contenir un namespace.
 * Par exemple:
 *
 \code
 * ARCANE_SERVICE_INTERFACE(Arcane::IUnitTest);
 \endcode
 *
 */
#define ARCANE_SERVICE_INTERFACE(ainterface)\
  Arcane::Internal::ServiceInterfaceRegisterer< ainterface >(#ainterface)

//! Enregistre une interface avec un nom de namespace.
#define ARCANE_SERVICE_INTERFACE_NS(ainterface_ns,ainterface) \
  Arcane::Internal::ServiceInterfaceRegisterer<ainterface_ns :: ainterface>(#ainterface_ns,#ainterface)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Service
 * \brief Macro pour enregistrer un service.
 *
 * L'appel est comme suit:
 *
 \code
 * ARCANE_REGISTER_SERVICE(ClassName,
 *                         ServiceProperty("ServiceName",where),
 *                         ARCANE_SERVICE_INTERFACE(InterfaceName1),);
 *                         ARCANE_SERVICE_INTERFACE(InterfaceName2),...);
 \endcode

 * Avec les paramètres suivants:
 * - \a ClassName est le nom de la classe du service,
 * - \a "ServiceName" est le nom du service.
 * - \a where est de type eServiceType et indique où le service peut être créé.
 * - \a InterfaceName est le nom de l'interface implémentée par le service. Il
 * est possible de spécifier plusieurs interfaces pour un même service.
 *
 * Par exemple, on peut avoir une utilisation comme suit:
 *
 \code
 * ARCANE_REGISTER_SERVICE(ThreadParallelSuperMng,
 *                         ServiceProperty("ThreadParallelSuperMng",ST_Application),
 *                         ARCANE_SERVICE_INTERFACE(IParallelSuperMng));
 \endcode
 *
 * \note Cette macro utilise un objet global pour enregistrer le service et
 * ne doit donc pas être utilisée dans un fichier qui peut appartenir à plusieurs
 * unités de compilation (par exemple il ne doit pas se trouve dans un fichier d'en-tête).
 */
#define ARCANE_REGISTER_SERVICE(aclass,a_service_property,...) \
namespace\
{\
  Arcane::IServiceInfo*\
  ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass) (const Arcane::ServiceProperty& property) \
  {\
    auto* si = Arcane::Internal::ServiceInfo::create(property,__FILE__,__LINE__); \
    Arcane::Internal::ServiceAllInterfaceRegisterer<aclass> :: registerToServiceInfo(si,__VA_ARGS__); \
    return si;\
  }\
}\
Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass) \
  (& ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass),a_service_property)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Macro pour enregistrer un service issu d'un fichier AXL.
 *
 * Cette macro est interne à Arcane et ne doit pas être utilisée directement
 */
#define ARCANE_REGISTER_AXL_SERVICE(aclass,a_service_properties) \
namespace\
{\
  Arcane::IServiceInfo*\
  ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass) (const Arcane::ServiceProperty& properties) \
  {\
    Arcane::ServiceInfo* si = Arcane::ServiceInfo::create(properties,__FILE__,__LINE__); \
    aclass :: fillServiceInfo< aclass >(si);                            \
    return si;\
  }\
}\
Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass) \
  (& ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass),a_service_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Les types et macros suivants sont obsolètes.
 *
 * A terme, seule la macro ARCANE_REGISTER_SERVICE sera utilisée.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistre un service de fabrique pour la classe \a aclass
 *
 * Le service est enregistré sous le nom \a aname et implémente
 * l'interface \a ainterface.
 *
 * \deprecated Utiliser ARCANE_REGISTER_SERVICE() à la place.
 */
#define ARCANE_REGISTER_APPLICATION_FACTORY(aclass,ainterface,aname) \
ARCANE_REGISTER_SERVICE ( aclass, Arcane::ServiceProperty(#aname,Arcane::ST_Application) ,\
                          ARCANE_SERVICE_INTERFACE(ainterface) )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistre un service de fabrique pour la classe \a aclass
 *
 * Le service est enregistré sous le nom \a aname et implémente
 * l'interface \a ainterface.
 *
 * \deprecated Utiliser ARCANE_REGISTER_SERVICE() à la place.
 */
#define ARCANE_REGISTER_SUB_DOMAIN_FACTORY(aclass,ainterface,aname)  \
ARCANE_REGISTER_SERVICE ( aclass, Arcane::ServiceProperty(#aname,Arcane::ST_SubDomain) ,\
                          ARCANE_SERVICE_INTERFACE(ainterface) )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistre un service de fabrique pour la classe \a aclass
 *
 * Le service est enregistré sous le nom \a aname et implémente
 * l'interface \a ainterface du namespace \a ainterface_ns.
 *
 * \deprecated Utiliser ARCANE_REGISTER_SERVICE() à la place.
 */
#define ARCANE_REGISTER_SUB_DOMAIN_FACTORY4(aclass,ainterface_ns,ainterface,aname) \
ARCANE_REGISTER_SERVICE ( aclass, Arcane::ServiceProperty(#aname,Arcane::ST_SubDomain) ,\
                          ARCANE_SERVICE_INTERFACE_NS(ainterface_ns,ainterface) )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistre un service de fabrique pour la classe \a aclass
 *
 * Le service est enregistré sous le nom \a aname et implémente
 * l'interface \a ainterface.
 *
 * \deprecated Utiliser ARCANE_REGISTER_SERVICE() à la place.
 */
#define ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(aclass,ainterface,aname) \
ARCANE_REGISTER_SERVICE ( aclass, Arcane::ServiceProperty(#aname,Arcane::ST_CaseOption) , \
                          ARCANE_SERVICE_INTERFACE(ainterface) )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistre un service de fabrique pour la classe \a aclass
 *
 * Le service est enregistré sous le nom \a aname et implémente
 * l'interface \a ainterface.
 *
 * \deprecated Utiliser ARCANE_REGISTER_SERVICE() à la place.
 */
#define ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY4(aclass,ainterface_ns,ainterface,aname) \
ARCANE_REGISTER_SERVICE ( aclass, Arcane::ServiceProperty(#aname,Arcane::ST_CaseOption) ,\
                          ARCANE_SERVICE_INTERFACE_NS(ainterface_ns,ainterface) )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

