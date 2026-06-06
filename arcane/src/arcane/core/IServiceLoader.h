// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceLoader.h                                            (C) 2000-2025 */
/*                                                                           */
/* Service and module loading interface.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICELOADER_H
#define ARCANE_CORE_ISERVICELOADER_H
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
 * \brief Service loading interface.
 */
class IServiceLoader
{
 public:

  //! Releases resources
  virtual ~IServiceLoader() = default;

 public:

  //! Type of a function that returns a factory for a given service.
  typedef IServiceFactory* (*CreateServiceFactoryFunc)(IServiceInfo*);

 public:

  //! Loads available application singleton and autoload services
  virtual void loadApplicationServices(IApplication*) = 0;

  //! Loads available session singleton and autoload services
  virtual void loadSessionServices(ISession*) = 0;

  //! Loads available subdomain singleton and autoload services
  virtual void loadSubDomainServices(ISubDomain* sd) = 0;

  /*!
   * \brief Loads the subdomain singleton service with name \a name.
   *
   * Returns \a true upon success and \a false if the singleton service
   * is not found.
   */
  virtual bool loadSingletonService(ISubDomain* sd, const String& name) = 0;

  /*!
   * \brief Loads modules in the subdomain \a sd.
   *
   * If \a all_modules is true, all modules are loaded; otherwise,
   * only modules with the 'autoload' attribute are loaded
   */
  virtual void loadModules(ISubDomain* sd, bool all_modules) = 0;

  //! Calls the initialization methods for module factories.
  virtual void initializeModuleFactories(ISubDomain* sd) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
