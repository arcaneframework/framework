// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFinder2.h                                            (C) 2000-2025 */
/*                                                                           */
/* Class to find a given service.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERVICEFINDER2_H
#define ARCANE_CORE_SERVICEFINDER2_H
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

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Utility class to find one or more services
 * implementing the \a InterfaceType interface.
 */
template <typename InterfaceType>
class ServiceFinderBase2T
{
 public:

  typedef IServiceFactory2T<InterfaceType> FactoryType;

 public:

  ServiceFinderBase2T(IApplication* app, const ServiceBuildInfoBase& sbi)
  : m_application(app)
  , m_service_build_info_base(sbi)
  {
  }

  virtual ~ServiceFinderBase2T() {}

 public:

  /*!
   * \brief Creates an instance of the service \a name.
   *
   * Returns null if no service with this name exists.
   *
   * \deprecated Use createReference() instead.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  virtual InterfaceType* create(const String& name)
  {
    return _create(name, m_service_build_info_base);
  }

  /*!
   * \brief Creates a reference to the service \a name.
   *
   * Returns a null reference if no service with this name exists.
   */
  virtual Ref<InterfaceType> createReference(const String& name)
  {
    return _createReference(name, m_service_build_info_base);
  }

  /*!
   * \brief Creates an instance of the service \a name for the mesh \a mesh.
   *
   * This is only valid for subdomain services. For others,
   * it has no effect.
   * The caller must destroy these services.
   * Returns null if no service with this name exists.
   *
   * \deprecated Use createReference() instead.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  virtual InterfaceType* create(const String& name, IMesh* mesh)
  {
    ISubDomain* sd = m_service_build_info_base.subDomain();
    if (!sd)
      return {};
    if (mesh)
      return _create(name, ServiceBuildInfoBase(sd, mesh));
    return _create(name, ServiceBuildInfoBase(sd));
  }

  /*!
   * \brief Creates a reference to the service \a name for the mesh \a mesh.
   *
   * This is only valid for subdomain services. For others,
   * it has no effect.
   * The caller must destroy these services.
   * Returns null if no service with this name exists.
   */
  virtual Ref<InterfaceType> createReference(const String& name, IMesh* mesh)
  {
    ISubDomain* sd = m_service_build_info_base.subDomain();
    if (!sd)
      return {};
    if (mesh)
      return _createReference(name, ServiceBuildInfoBase(sd, mesh));
    return _createReference(name, ServiceBuildInfoBase(sd));
  }

  /*!
   * \brief Singleton instance of the service having the \a InterfaceType interface.
   *
   * Returns null if no service is found
   */
  virtual InterfaceType* getSingleton()
  {
    IServiceMng* sm = m_service_build_info_base.serviceParent()->serviceMng();
    SingletonServiceInstanceCollection singleton_services = sm->singletonServices();
    for (typename SingletonServiceInstanceCollection::Enumerator i(singleton_services); ++i;) {
      ISingletonServiceInstance* ssi = (*i).get();
      if (ssi) {
        for (typename ServiceInstanceCollection::Enumerator k(ssi->interfaceInstances()); ++k;) {
          IServiceInstance* sub_isi = (*k).get();
          auto m = dynamic_cast<IServiceInstanceT<InterfaceType>*>(sub_isi);
          if (m)
            return m->instance().get();
        }
      }
    }
    return nullptr;
  }

  /*!
   * \brief Creates an instance of every service that implements \a InterfaceType.
   *
   * The caller must destroy these services via the call to 'operator delete'.
   *
   * \deprecated Use the overload taking an array of references instead.
   */
  ARCCORE_DEPRECATED_2019("Use createAll(Array<ServiceRef<InterfaceType>>&) instead")
  virtual void createAll(Array<InterfaceType*>& instances)
  {
    _createAll(instances, m_service_build_info_base);
  }

  /*!
   * \brief Creates an instance of every service that implements \a InterfaceType.
   */
  virtual UniqueArray<Ref<InterfaceType>> createAll()
  {
    return _createAll(m_service_build_info_base);
  }

 public:

  SharedArray<FactoryType*> factories()
  {
    SharedArray<FactoryType*> m_factories;
    for (typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j;) {
      IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast<IServiceFactory2T<InterfaceType>*>(sf2);
      //m_application->traceMng()->info() << " FOUND sf2=" << sf2 << " M=" << m;
      if (m) {
        m_factories.add(m);
      }
    }
    return m_factories.constView();
  }

  void getServicesNames(Array<String>& names) const
  {
    for (typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j;) {
      IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* true_factory = dynamic_cast<IServiceFactory2T<InterfaceType>*>(sf2);
      if (true_factory) {
        IServiceInfo* si = sf2->serviceInfo();
        names.add(si->localName());
      }
    }
  }

 protected:

  InterfaceType* _create(const String& name, const ServiceBuildInfoBase& sbib)
  {
    return _createReference(name, sbib)._release();
  }

  Ref<InterfaceType> _createReference(const String& name, const ServiceBuildInfoBase& sbib)
  {
    for (typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j;) {
      Internal::IServiceFactory2* sf2 = *j;
      IServiceInfo* s = sf2->serviceInfo();
      if (s->localName() != name)
        continue;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast<IServiceFactory2T<InterfaceType>*>(sf2);
      //m_application->traceMng()->info() << " FOUND sf2=" << sf2 << " M=" << m;
      if (m) {
        Ref<InterfaceType> tt = m->createServiceReference(sbib);
        if (!tt.isNull())
          return tt;
      }
    }
    return {};
  }

  void _createAll(Array<InterfaceType*>& instances, const ServiceBuildInfoBase& sbib)
  {
    UniqueArray<Ref<InterfaceType>> ref_instances = _createAll(sbib);
    for (auto& x : ref_instances)
      instances.add(x._release());
  }

  UniqueArray<Ref<InterfaceType>> _createAll(const ServiceBuildInfoBase& sbib)
  {
    UniqueArray<Ref<InterfaceType>> instances;
    for (typename ServiceFactory2Collection::Enumerator j(this->m_application->serviceFactories2()); ++j;) {
      Internal::IServiceFactory2* sf2 = *j;
      IServiceFactory2T<InterfaceType>* m = dynamic_cast<IServiceFactory2T<InterfaceType>*>(sf2);
      if (m) {
        Ref<InterfaceType> tt = m->createServiceReference(sbib);
        if (tt.get()) {
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

} // namespace Arcane::Internal

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utility class to find one or more services
 * implementing the \a InterfaceType interface.
 * \deprecated This class should no longer be used directly.
 * Use ServiceBuilder instead.
 */
template <typename InterfaceType, typename ParentType>
class ServiceFinder2T
: public Internal::ServiceFinderBase2T<InterfaceType>
{
 public:

  ServiceFinder2T(IApplication* app, ParentType* parent)
  : Internal::ServiceFinderBase2T<InterfaceType>(app, ServiceBuildInfoBase(parent))
  {
  }

  ~ServiceFinder2T() {}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
