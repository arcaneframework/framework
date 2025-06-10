// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceLoader.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface de chargement des services et modules.                          */
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
 * \brief Interface de chargement des services.
 */
class IServiceLoader
{
 public:

  //! Libère les ressources
  virtual ~IServiceLoader() = default;

 public:
	
  //! Type d'une fonction retournant une fabrique pour un service donné.
  typedef IServiceFactory* (*CreateServiceFactoryFunc)(IServiceInfo*);

 public:

  //! Charge les services singletons et autoload applicatifs disponibles
  virtual void loadApplicationServices(IApplication*) =0;

  //! Charge les services singletons et autoload de session disponibles
  virtual void loadSessionServices(ISession*) =0;

  //! Charge les services singletons et autoload de sous-domaine disponibles
  virtual void loadSubDomainServices(ISubDomain* sd) =0;

  /*!
   * \brief Charge le service singleton de sous-domaine de nom \a name.
   *
   * Retourne \a true en cas de succès et \a false si le service singleton
   * n'est pas trouvé.
   */
  virtual bool loadSingletonService(ISubDomain* sd,const String& name) =0;

  /*!
   * \brief Charge les modules dans le sous-domaine \a sd.
   *
   * Si \a all_modules est vrai, tous les modules sont chargés, sinon,
   * seul les modules avec l'attribut 'autoload' sont chargés
   */
  virtual void loadModules(ISubDomain* sd,bool all_modules) =0;

  //! Appel les méthodes d'initialisation des fabriques des modules.
  virtual void initializeModuleFactories(ISubDomain* sd) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
