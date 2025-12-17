// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceAndModuleFactoryMng.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un gestionnaire de fabriques de services et modules.          */
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
 * \brief Interface d'un gestionnaire de fabriques de services et modules.
 */
class ARCANE_CORE_EXPORT IServiceAndModuleFactoryMng
{
 public:

  virtual ~IServiceAndModuleFactoryMng() = default; //!< Libère les ressources.

 public:

  /*!
   * \brief Créé toutes les fabriques associées à des ServiceRegisterer.
   *
   * Cette méthode peut être appelée plusieurs fois si on souhaite
   * enregistrer les nouveaux services disponibles par exemple
   * après un chargement dynamique de bibliothèque.
   */
  virtual void createAllServiceRegistererFactories() = 0;

 public:

  //! Liste des informations sur les fabriques des services
  virtual ServiceFactoryInfoCollection serviceFactoryInfos() const = 0;
  //! Liste des informations sur les fabriques des modules
  virtual ServiceFactory2Collection serviceFactories2() const = 0;
  //! Liste des fabriques de service.
  virtual ModuleFactoryInfoCollection moduleFactoryInfos() const = 0;

  /*!
   * \brief Ajoute la fabrique de service \a sfi.
   * \a sfi ne doit pas être détruit tant que cette instance est utilisée.
   * Si \a sfi est déjà enregistréé, aucune opération n'est effectuée.
   */
  virtual void addGlobalFactory(IServiceFactoryInfo* sfi) = 0;

  /*!
   * \brief Ajoute la fabrique de module \a mfi.
   * \a mfi ne doit pas être détruit tant que cette instance est utilisée.
   * Si \a mfi est déjà enregistréé, aucune opération n'est effectuée.
   */
  virtual void addGlobalFactory(IModuleFactoryInfo* mfi) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
