// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceAndModuleFactoryMng.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a service and module factory manager.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEANDMODULEFACTORYMNG_H
#define ARCANE_CORE_ISERVICEANDMODULEFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a service and module factory manager.
 */
class ARCANE_CORE_EXPORT IServiceAndModuleFactoryMng
{
 public:

  virtual ~IServiceAndModuleFactoryMng() = default; //!< Releases resources.

 public:

  /*!
   * \brief Creates all factories associated with a ServiceRegisterer.
   *
   * This method can be called multiple times if you wish to register new available services, for example,
   * after a dynamic library load.
   */
  virtual void createAllServiceRegistererFactories() = 0;

 public:

  //! List of information about service factories
  virtual ServiceFactoryInfoCollection serviceFactoryInfos() const = 0;
  //! List of information about module factories
  virtual ServiceFactory2Collection serviceFactories2() const = 0;
  //! List of service factories.
  virtual ModuleFactoryInfoCollection moduleFactoryInfos() const = 0;

  /*!
   * \brief Adds the service factory \a sfi.
   * \a sfi must not be destroyed while this instance is in use.
   * If \a sfi is already registered, no operation is performed.
   */
  virtual void addGlobalFactory(IServiceFactoryInfo* sfi) = 0;

  /*!
   * \brief Adds the module factory \a mfi.
   * \a mfi must not be destroyed while this instance is in use.
   * If \a mfi is already registered, no operation is performed.
   */
  virtual void addGlobalFactory(IModuleFactoryInfo* mfi) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
