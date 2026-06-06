// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceFactory.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Factory for services/modules.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/IServiceInfo.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceInstance.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ServiceFactory.h
 *
 * \brief This file contains the various service factories
 * and macros for registering services.
 *
 * Most types in this file are internal to Arcane. The only element
 * useful for a user is the ARCANE_REGISTER_SERVICE() macro, which
 * allows a service to be registered.
 */

/*!
 * \file ServiceProperty.h
 *
 * \brief This file contains the various types and classes
 * for specifying service properties.
 */


/*!
 * \defgroup Service Service
 *
 * \brief Collection of types used in service management.
 *
 * Most user services are subdomain services
 * and derive indirectly from the BasicService class. Generally, a service is defined in an
 * AXL file, and the tool \a axl2cc allows generating the base class
 * of a service from this AXL file. For more
 * information, refer to section \ref arcanedoc_core_types_service.
 *
 * Nevertheless, it is possible to have services without an AXL file. In this case, registering a service so that it is
 * recognized by Arcane is done via the ARCANE_REGISTER_SERVICE() macro.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Singleton service instances.
 *
 * Singleton services can implement multiple interfaces.
 * There is therefore one IServiceInstance instance per interface plus an instance
 * for the service itself. Since all these instances reference the
 * same service, care must be taken not to destroy the service more than once.
 */
class SingletonServiceFactoryBase::ServiceInstance
: public ISingletonServiceInstance
, public IServiceInstanceAdder
{
 public:
  ServiceInstance(IServiceInfo* si)
  : m_service_info(si){}
  ~ServiceInstance()
  {
    destroyInstance();
  }
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  ServiceInstanceCollection interfaceInstances() override { return m_instances; }
  void destroyInstance()
  {
    m_true_instance.reset();
    m_instances.clear();
  }
  IServiceInfo* serviceInfo() const override { return m_service_info; }
  void setTrueInstance(ServiceInstanceRef si) { m_true_instance = si; }
 public:
  void addInstance(ServiceInstanceRef instance) override
  {
    m_instances.add(instance);
  }
 private:
  IServiceInfo* m_service_info;
  List<ServiceInstanceRef> m_instances;
  ServiceInstanceRef m_true_instance;
  std::atomic<Int32> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Created a singleton service
Ref<ISingletonServiceInstance> SingletonServiceFactoryBase::
createSingletonServiceInstance(const ServiceBuildInfoBase& sbib)
{
  auto x = new ServiceInstance(m_service_info);
  IServiceInstanceAdder* sia = x;
  ServiceInstanceRef si = _createInstance(sbib,sia);
  x->setTrueInstance(si);
  return makeRef<ISingletonServiceInstance>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractServiceFactory::
addReference()
{
  ++m_nb_ref;
}

void AbstractServiceFactory::
removeReference()
{
  // Decrements and returns the previous value.
  // If it is 1, it means there are no more references
  // to the object and it must be destroyed.
  Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
  if (v==1)
    delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
