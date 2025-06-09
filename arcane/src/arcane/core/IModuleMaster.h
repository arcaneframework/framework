// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleMaster.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface du module Maître.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMODULEMASTER_H
#define ARCANE_CORE_IMODULEMASTER_H
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
 * \brief Interface du module principal.
 * 
 * Le module principal est le module encadrant les différentes actions des points d'entrée.
 * Voir l'implémentation \a ModuleMaster pour plus de détails.
 */
class ARCANE_CORE_EXPORT IModuleMaster
{
 public:

  //! Destructeur.
  /*! Libère les ressources */
  virtual ~IModuleMaster() {}

 public:

  //! Création d'une instance de IModuleMaster
  /*! Actuellement implémenté dans \a ModuleMaster */
  static IModuleMaster* createDefault(const ModuleBuildInfo&);

 public:

  //! Retourne les options de ce module
  virtual CaseOptionsMain* caseoptions() = 0;

  //! Conversion en module standard
  /*! Le succès de la conversion est liée à l'implémentation de \a IModuleMaster en tant que \a IModule */
  virtual IModule* toModule() = 0;

  //! Accès aux variables 'communes' partagés entre tout service et module
  virtual CommonVariables* commonVariables() = 0;

  //! Ajoute le service de boucle en temps
  virtual void addTimeLoopService(ITimeLoopService* tls) = 0;

  /*!
   * \brief Sort les courbes classiques.
   *
   * Cet appel ajoute dans le ITimeHistoryMng les courbes classiques
   * (telles que CPUTime, ElapsedTime, TotalMemory, ...) pour l'itération
   * courante. Par défaut, si cette fonction n'est pas appelée, les
   * sorties se font à la fin de l'itération.
   */
  virtual void dumpStandardCurves() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

