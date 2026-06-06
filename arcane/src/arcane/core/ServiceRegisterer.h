// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceRegisterer.h                                         (C) 2000-2025 */
/*                                                                           */
/* Singleton allowing service registration.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERCVICEREGISTERER_H
#define ARCANE_CORE_SERCVICEREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ServiceProperty.h"
#include "arcane/core/ModuleProperty.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service and module registrar
 *
 * This class implements the Singleton pattern for a given service.
 * 
 * It allows declaring a global variable that automatically registers
 * the IServiceFactory of the desired service. This class is not used
 * directly but through the ARCANE_DEFINE_SERVICE(name) macro.
 * 
 * Since this class is used with global instances, they are
 * constructed before the code enters main(). Therefore, extreme care
 * must be taken not to use any objects or perform allocations (in
 * particular, the service name must be a standard const char*). For
 * this purpose, the list of registered services is maintained in a
 * linked list, and each instance contains pointers to the next and
 * previous members of the list. The first element of the list is
 * obtained by calling ServiceRegisterer::firstService().
 */
class ARCANE_CORE_EXPORT ServiceRegisterer
{
 public:

  typedef IModuleFactoryInfo* (*ModuleFactoryWithPropertyFunc)(const ModuleProperty& properties);
  typedef IServiceInfo* (*ServiceInfoWithPropertyCreateFunc)(const ServiceProperty& properties);

 public:

  /*!
   * \brief Creates a registrar for the service \a name and the function \a func.
   *
   * This constructor is used to register a service.
   */
  ServiceRegisterer(ServiceInfoWithPropertyCreateFunc func, const ServiceProperty& properties) ARCANE_NOEXCEPT;

  /*!
   * \brief Creates a registrar for the module \a name with properties \a properties.
   *
   * This constructor is used to register a module.
   */
  ServiceRegisterer(ModuleFactoryWithPropertyFunc func, const ModuleProperty& properties) ARCANE_NOEXCEPT;

 public:

  /*!
   * \brief Creation function for the 'ServiceInfo' instance if it is a service.
   *
   * This pointer may be null if it is not a service, in which case
   * infoCreatorFunction() must be used.
   */
  ServiceInfoWithPropertyCreateFunc infoCreatorWithPropertyFunction() { return m_info_function_with_property; }

  /*!
   * \brief Creation function for the factory if it is a module.
   *
   * This pointer may be null if it is not a module, in which case
   * infoCreatorFunction() must be used.
   */
  ModuleFactoryWithPropertyFunc moduleFactoryWithPropertyFunction() { return m_module_factory_with_property_functor; }

  //! Service name
  const char* name() { return m_name; }

  /*!
   * \brief Service properties.
   *
   * \deprecated Use \a serviceProperty() instead
   */
  ARCANE_DEPRECATED_260 const ServiceProperty& property() const { return m_service_property; }

  //! Properties in the case of a service
  const ServiceProperty& serviceProperty() const { return m_service_property; }

  //! Properties in the case of a module
  const ModuleProperty& moduleProperty() const { return m_module_property; }

  //! Previous service (0 if the first)
  ServiceRegisterer* previousService() const { return m_previous; }

  //! Next service (0 if the last)
  ServiceRegisterer* nextService() const { return m_next; }

 private:

  //! Positions the previous service
  /*! Used internally to build the service chain */
  void setPreviousService(ServiceRegisterer* s) { m_previous = s; }

  //! Positions the next service
  /*! Used internally to build the service chain */
  void setNextService(ServiceRegisterer* s) { m_next = s; }

 public:

  //! Access to the first element of the service registrar chain
  static ServiceRegisterer* firstService();

  //! Number of service registrars in the chain
  static Integer nbService();

 private:

  //! Function to create the IModuleFactory
  ModuleFactoryWithPropertyFunc m_module_factory_with_property_functor = nullptr;
  //! Function to create the IServiceInfo
  ServiceInfoWithPropertyCreateFunc m_info_function_with_property = nullptr;
  //! Service name
  const char* m_name = nullptr;
  //! Service properties
  ServiceProperty m_service_property;
  //! Module properties
  ModuleProperty m_module_property;
  //! Previous service
  ServiceRegisterer* m_previous = nullptr;
  //! Next service
  ServiceRegisterer* m_next = nullptr;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
