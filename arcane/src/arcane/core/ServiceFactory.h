// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFactory.h                                            (C) 2000-2025 */
/*                                                                           */
/* Service manufacturing.                                                    */
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
 * \brief Base class for a service instance in C#.
 */
class DotNetServiceInstance
: public IServiceInstance
{
 public:

  DotNetServiceInstance(IServiceInfo* si)
  : m_service_info(si)
  , m_handle(nullptr)
  {}

 public:

  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
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
 * \brief Service instance.
 */
template <typename InterfaceType>
class ServiceInstanceT
: public IServiceInstanceT<InterfaceType>
{
 public:

  ServiceInstanceT(Ref<InterfaceType> i, IServiceInfo* si)
  : m_instance(i)
  , m_service_info(si)
  {}

 public:

  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
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
 * \brief Information about the manufacturing of a service or a module.
 */
class ARCANE_CORE_EXPORT ServiceFactoryInfo
: public IServiceFactoryInfo
{
 public:

  explicit ServiceFactoryInfo(IServiceInfo* si)
  : m_service_info(si)
  , m_is_autoload(false)
  , m_is_singleton(false)
  {}
  ~ServiceFactoryInfo() override {}

 public:

  IServiceInfo* serviceInfo() const override { return m_service_info; }

  bool isAutoload() const override { return m_is_autoload; }
  bool isSingleton() const override { return m_is_singleton; }

  virtual bool isModule() const { return false; }
  virtual void initializeModuleFactory(ISubDomain*) {}
  virtual IModule* createModule(ISubDomain*, IMesh*) { return nullptr; }

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
  void initProperties() {}

 private:

  IServiceInfo* m_service_info;
  bool m_is_autoload;
  bool m_is_singleton;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a factory function (functor) that creates a service
 * instance corresponding to the \a InterfaceType interface.
 */
template <typename InterfaceType>
class IServiceInterfaceFactory
{
 public:

  virtual ~IServiceInterfaceFactory() = default;

 public:

  //! Creates an instance of the service.
  virtual Ref<InterfaceType> createReference(const ServiceBuildInfo& sbi) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Service factory.
 *
 * This class implements IServiceFactory2 and IServiceFactory2T for
 * the \a InterfaceType interface.
 */
template <typename InterfaceType>
class ServiceFactory2TV2
: public IServiceFactory2T<InterfaceType>
{
 public:

  ServiceFactory2TV2(IServiceInfo* si, IServiceInterfaceFactory<InterfaceType>* sub_factory)
  : m_service_info(si)
  , m_sub_factory(sub_factory)
  , m_type_flags(si->usageType())
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
    InterfaceType* it = m_sub_factory->createInstance(ServiceBuildInfo(m_service_info, sbib));
    return it;
  }

  Ref<InterfaceType> _createReference(const ServiceBuildInfoBase& sbib)
  {
    return m_sub_factory->createReference(ServiceBuildInfo(m_service_info, sbib));
  }

  ServiceInstanceRef _create(Ref<InterfaceType> it)
  {
    IServiceInstance* x = (!it) ? nullptr : new ServiceInstanceT<InterfaceType>(it, m_service_info);
    return ServiceInstanceRef::createRef(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory for the \a ServiceType service for the \a InterfaceType interface.
 */
template <typename ServiceType, typename InterfaceType>
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
  virtual void addInstance(ServiceInstanceRef instance) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for factories for singleton services.
 *
 * You must derive from this class and implement _createInstance().
 */
class ARCANE_CORE_EXPORT SingletonServiceFactoryBase
: public ISingletonServiceFactory
{
 public:

  class ServiceInstance;

 public:

  explicit SingletonServiceFactoryBase(IServiceInfo* si)
  : m_service_info(si)
  {}

 public:

  //! Creates a singleton service instance
  Ref<ISingletonServiceInstance> createSingletonServiceInstance(const ServiceBuildInfoBase& sbib) override;

  //! Returns the IServiceInfo associated with this factory.
  IServiceInfo* serviceInfo() const override { return m_service_info; }

 protected:

  virtual ServiceInstanceRef _createInstance(const ServiceBuildInfoBase& sbi, IServiceInstanceAdder* instance_adder) = 0;

 private:

  IServiceInfo* m_service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory for the singleton service of type \a ServiceType
 * implementing the \a Interfaces interfaces.
 */
template <typename ServiceType, typename... Interfaces>
class SingletonServiceFactory
: public SingletonServiceFactoryBase
{
  /*!
   * \brief Utility class allowing the creation of an IServiceInstance
   * for each interface in \a Interfaces.
   */
  class Helper
  {
   public:

    Helper(ServiceType* service, IServiceInfo* si, IServiceInstanceAdder* adder)
    : m_service(service)
    , m_service_info(si)
    , m_adder(adder)
    {}

   private:

    //! Overload for 1 interface
    template <typename InterfaceType> void _create()
    {
      InterfaceType* x = m_service;
      // ATTENTION: the following reference must not destroy 'm_service' because
      // the latter already has a ServiceReference that was constructed during
      // the call to _createInstance().
      auto x_ref = Ref<InterfaceType>::_createNoDestroy(x);
      auto instance = new ServiceInstanceT<InterfaceType>(x_ref, m_service_info);
      // TODO: indicate that the reference should not be destroyed.
      auto instance_ref = ServiceInstanceRef::createRef(instance);
      m_adder->addInstance(instance_ref);
    }
    //! Overload for 2 or more interfaces
    template <typename I1, typename I2, typename... OtherInterfaces>
    void _create()
    {
      _create<I1>();
      // Apply recursion to the remaining types
      _create<I2, OtherInterfaces...>();
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

  explicit SingletonServiceFactory(IServiceInfo* si)
  : SingletonServiceFactoryBase(si)
  {}

 protected:

  ServiceInstanceRef _createInstance(const ServiceBuildInfoBase& sbib, IServiceInstanceAdder* instance_adder) override
  {
    ServiceBuildInfo sbi(serviceInfo(), sbib);
    ServiceType* st = new ServiceType(sbi);
    st->build();
    auto st_ref = makeRef(st);
    Helper ssf(st, serviceInfo(), instance_adder);
    ssf.createInterfaceInstances();
    IServiceInstance* si = new ServiceInstanceT<ServiceType>(st_ref, serviceInfo());
    return ServiceInstanceRef::createRef(si);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class allowing the registration of a factory for a service
 * implementing the \a InterfaceType interface.
 */
template <typename InterfaceType>
class ServiceInterfaceRegisterer
{
 public:

  typedef InterfaceType Interface;

  explicit ServiceInterfaceRegisterer(const char* name)
  : m_name(name)
  , m_namespace_name(nullptr)
  {
  }

  ServiceInterfaceRegisterer(const char* namespace_name, const char* name)
  : m_name(name)
  , m_namespace_name(namespace_name)
  {
  }

 public:

  //! Registers in \a si a factory to create an instance of the service \a ServiceType
  template <typename ServiceType> void
  registerToServiceInfo(ServiceInfo* si) const
  {
    IServiceInterfaceFactory<InterfaceType>* factory = new ServiceInterfaceFactory<ServiceType, InterfaceType>();
    if (m_namespace_name)
      si->addImplementedInterface(String(m_namespace_name) + String("::") + String(m_name));
    else
      si->addImplementedInterface(m_name);
    si->addFactory(new ServiceFactory2TV2<InterfaceType>(si, factory));
  }

 private:

  const char* m_name;
  const char* m_namespace_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class allowing the creation and registration of factories for a service.
 */
template <typename ServiceType>
class ServiceAllInterfaceRegisterer
{
 private:

  //! Overload for 1 interface
  template <typename InterfaceType> static void
  _create(ServiceInfo* si, const InterfaceType& i1)
  {
    i1.template registerToServiceInfo<ServiceType>(si);
  }
  //! Overload for 2 or more interfaces
  template <typename I1, typename I2, typename... OtherInterfaces>
  static void _create(ServiceInfo* si, const I1& i1, const I2& i2, const OtherInterfaces&... args)
  {
    _create<I1>(si, i1);
    // Apply recursion to the remaining types
    _create<I2, OtherInterfaces...>(si, i2, args...);
  }

 public:

  //! Registers the factories for the \a Interfaces interfaces in the service
  template <typename... Interfaces> static void
  registerToServiceInfo(ServiceInfo* si, const Interfaces&... args)
  {
    si->setSingletonFactory(new Internal::SingletonServiceFactory<ServiceType, typename Interfaces::Interface...>(si));
    _create(si, args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to declare an interface when registering a service.
 *
 * This macro is used within the ARCANE_REGISTER_SERVICE macro.
 *
 * The call is as follows:
 * \code
 * ARCANE_SERVICE_INTERFACE(ainterface)
 \endcode
 *
 * \a ainterface is the name of the interface (without quotes). It
 * may contain a namespace.
 * For example:
 * \code
 * ARCANE_SERVICE_INTERFACE(Arcane::IUnitTest);
 \endcode
 *
 */
#define ARCANE_SERVICE_INTERFACE(ainterface) \
  Arcane::Internal::ServiceInterfaceRegisterer<ainterface>(#ainterface)

//! Registers an interface with a namespace name.
#define ARCANE_SERVICE_INTERFACE_NS(ainterface_ns, ainterface) \
  Arcane::Internal::ServiceInterfaceRegisterer<ainterface_ns ::ainterface>(#ainterface_ns, #ainterface)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Service
 * \brief Macro for registering a service.
 *
 * The call is as follows:
 * \code
 * ARCANE_REGISTER_SERVICE(ClassName,
 *                         ServiceProperty("ServiceName",where),
 *                         ARCANE_SERVICE_INTERFACE(InterfaceName1),);
 *                         ARCANE_SERVICE_INTERFACE(InterfaceName2),...);
 \endcode

 * With the following parameters:
 * - \a ClassName is the name of the service class,
 * - \a "ServiceName" is the name of the service.
 * - \a where is of type eServiceType and indicates where the service can be created.
 * - \a InterfaceName is the name of the interface implemented by the service. It
 * is possible to specify multiple interfaces for the same service.
 *
 * For example, usage can be as follows:
 * \code
 * ARCANE_REGISTER_SERVICE(ThreadParallelSuperMng,
 *                         ServiceProperty("ThreadParallelSuperMng",ST_Application),
 *                         ARCANE_SERVICE_INTERFACE(IParallelSuperMng));
 \endcode
 *
 * \note This macro uses a global object to register the service and
 * should therefore not be used in a file that may belong to multiple
 * compilation units (for example, it should not be in a header file).
 */
#define ARCANE_REGISTER_SERVICE(aclass, a_service_property, ...) \
  namespace \
  { \
    Arcane::IServiceInfo* \
    ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass)(const Arcane::ServiceProperty& property) \
    { \
      auto* si = Arcane::Internal::ServiceInfo::create(property, __FILE__, __LINE__); \
      Arcane::Internal::ServiceAllInterfaceRegisterer<aclass>::registerToServiceInfo(si, __VA_ARGS__); \
      return si; \
    } \
  } \
  Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass)(&ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass), a_service_property)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Macro for registering a service derived from an AXL file.
 *
 * This macro is internal to Arcane and should not be used directly
 */
#define ARCANE_REGISTER_AXL_SERVICE(aclass, a_service_properties) \
  namespace \
  { \
    Arcane::IServiceInfo* \
    ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass)(const Arcane::ServiceProperty& properties) \
    { \
      Arcane::ServiceInfo* si = Arcane::ServiceInfo::create(properties, __FILE__, __LINE__); \
      aclass ::fillServiceInfo<aclass>(si); \
      return si; \
    } \
  } \
  Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass)(&ARCANE_JOIN_WITH_LINE(arcaneCreateServiceInfo##aclass), a_service_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * The following types and macros are obsolete.
 *
 * In the future, only the ARCANE_REGISTER_SERVICE macro will be used.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers a factory service for the class \a aclass
 *
 * The service is registered under the name \a aname and implements
 * the \a ainterface interface.
 *
 * \deprecated Use ARCANE_REGISTER_SERVICE() instead.
 */
#define ARCANE_REGISTER_APPLICATION_FACTORY(aclass, ainterface, aname) \
  ARCANE_REGISTER_SERVICE(aclass, Arcane::ServiceProperty(#aname, Arcane::ST_Application), \
                          ARCANE_SERVICE_INTERFACE(ainterface))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers a factory service for the class \a aclass
 *
 * The service is registered under the name \a aname and implements
 * the \a ainterface interface.
 *
 * \deprecated Use ARCANE_REGISTER_SERVICE() instead.
 */
#define ARCANE_REGISTER_SUB_DOMAIN_FACTORY(aclass, ainterface, aname) \
  ARCANE_REGISTER_SERVICE(aclass, Arcane::ServiceProperty(#aname, Arcane::ST_SubDomain), \
                          ARCANE_SERVICE_INTERFACE(ainterface))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers a factory service for the class \a aclass
 *
 * The service is registered under the name \a aname and implements
 * the \a ainterface interface from the namespace \a ainterface_ns.
 *
 * \deprecated Use ARCANE_REGISTER_SERVICE() instead.
 */
#define ARCANE_REGISTER_SUB_DOMAIN_FACTORY4(aclass, ainterface_ns, ainterface, aname) \
  ARCANE_REGISTER_SERVICE(aclass, Arcane::ServiceProperty(#aname, Arcane::ST_SubDomain), \
                          ARCANE_SERVICE_INTERFACE_NS(ainterface_ns, ainterface))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers a factory service for the class \a aclass
 *
 * The service is registered under the name \a aname and implements
 * the \a ainterface interface.
 *
 * \deprecated Use ARCANE_REGISTER_SERVICE() instead.
 */
#define ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(aclass, ainterface, aname) \
  ARCANE_REGISTER_SERVICE(aclass, Arcane::ServiceProperty(#aname, Arcane::ST_CaseOption), \
                          ARCANE_SERVICE_INTERFACE(ainterface))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers a factory service for the class \a aclass
 *
 * The service is registered under the name \a aname and implements
 * the \a ainterface interface.
 *
 * \deprecated Use ARCANE_REGISTER_SERVICE() instead.
 */
#define ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY4(aclass, ainterface_ns, ainterface, aname) \
  ARCANE_REGISTER_SERVICE(aclass, Arcane::ServiceProperty(#aname, Arcane::ST_CaseOption), \
                          ARCANE_SERVICE_INTERFACE_NS(ainterface_ns, ainterface))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
