// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IService.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of a service.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICE_H
#define ARCANE_CORE_ISERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/ExternalRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service.
 *
 * The instances returned by serviceInfo() and serviceParent() are the
 * property of the application (IApplication interface) and must never
 * be modified or destroyed.
 *
 * \deprecated
 */
class ARCANE_CORE_EXPORT IService
{
 protected:

  //! Constructor
  IService() {}

 public:

  virtual ~IService() {} //!< Releases resources

 public:

  //! Parent of this service
  virtual IBase* serviceParent() const = 0;

  //! Interface of this service (normally this)
  virtual IService* serviceInterface() = 0;

  //! Service information
  virtual IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service instance.
 */
class ARCANE_CORE_EXPORT IServiceInstance
{
  friend class Ref<IServiceInstance>;

 protected:

  virtual ~IServiceInstance() = default;

 public:

  //! Adds a reference.
  virtual void addReference() = 0;
  //! Removes a reference.
  virtual void removeReference() = 0;
  virtual IServiceInfo* serviceInfo() const = 0;
  //! \internal
  virtual Internal::ExternalRef _internalDotNetHandle() const { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a singleton service instance.
 */
class ARCANE_CORE_EXPORT ISingletonServiceInstance
: public IServiceInstance
{
 public:

  //! List of instances of interfaces implemented by the singleton
  virtual ServiceInstanceCollection interfaceInstances() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Typed interface managing a service instance.
 */
template <typename InterfaceType>
class IServiceInstanceT
: public IServiceInstance
{
 public:

  virtual Ref<InterfaceType> instance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
