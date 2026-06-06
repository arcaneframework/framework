// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceFactory.h                                           (C) 2000-2025 */
/*                                                                           */
/* Service manufacturing interface.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEFACTORY_H
#define ARCANE_CORE_ISERVICEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/core/ArcaneTypes.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about a service factory.
 *
 * This interface contains the necessary information about a service factory.
 *
 * Generally, instances of this class are either created by Arcane
 * from an axl file, or using one of the service factory macros
 * (defined in ServiceFactory.h).
 *
 * The list of interfaces supported by the service and the
 * associated factories are described in serviceInfo().
 */
class ARCANE_CORE_EXPORT IServiceFactoryInfo
{
 public:

  //! Release resources
  virtual ~IServiceFactoryInfo() {}

 public:

  //! true if the service is a module and must be loaded automatically
  //TODO: check if autoload is still useful for these services.
  virtual bool isAutoload() const = 0;
  //! true if the service is a singleton service (a single instance)
  virtual bool isSingleton() const = 0;

 public:

  /*! \brief Information about the service that can be created by this factory.
   *
   * The returned instance remains the property of the application that created it
   * and must neither be modified nor destroyed.
   */
  virtual IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief internal.
 * \brief Interface for a service factory (new version).
 *
 * This class uses a ReferenceCounter to manage its destruction.
 */
class ARCANE_CORE_EXPORT IServiceFactory2
{
 protected:

  virtual ~IServiceFactory2() = default;

 public:

  //! Add a reference.
  virtual void addReference() = 0;
  //! Remove a reference.
  virtual void removeReference() = 0;

 public:

  //! Create a service instance from the info in \a sbi.
  virtual ServiceInstanceRef createServiceInstance(const ServiceBuildInfoBase& sbi) = 0;

  //! Returns the IServiceInfo associated with this factory.
  virtual IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief internal.
 * \brief Base class for a service factory.
 *
 * This class uses a ReferenceCounter to manage its destruction.
 */
class ARCANE_CORE_EXPORT AbstractServiceFactory
: public IServiceFactory2
{
 protected:

  AbstractServiceFactory()
  : m_nb_ref(0)
  {}

 public:

  void addReference() override;
  void removeReference() override;

 private:

  std::atomic<Int32> m_nb_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory for a service implementing the \a InterfaceType interface.
 */
template <typename InterfaceType>
class IServiceFactory2T
: public AbstractServiceFactory
{
 public:

  virtual Ref<InterfaceType> createServiceReference(const ServiceBuildInfoBase& sbi) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief internal.
 * \brief Factory for a singleton service.
 * A singleton service is created only once but can have multiple
 * interfaces, and therefore there are as many IServiceInstance objects as interfaces
 * implemented by the service.
 *
 * The \a createSingletonServiceInstance() method allows creating the singleton service instance
 * as well as the IServiceInstance for each implemented interface.
 */
class ARCANE_CORE_EXPORT ISingletonServiceFactory
{
 public:

  virtual ~ISingletonServiceFactory() = default;

  //! Create an instance of a singleton service.
  virtual Ref<ISingletonServiceInstance>
  createSingletonServiceInstance(const ServiceBuildInfoBase& sbi) = 0;

  //! Returns the IServiceInfo associated with this factory.
  virtual IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
