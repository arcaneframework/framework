// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSimpleExecutor.h                                      (C) 2000-2026 */
/*                                                                           */
/* Classe permettant d'exécuter du code directement via Arcane.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNALINFOSDUMPER_H
#define ARCANE_IMPL_INTERNALINFOSDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class ApplicationInfo;
class ApplicationBuildInfo;
class CommandLineArguments;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant d'exécuter directement du code sans
 * passer par la boucle en temps.
 *
 * Une seule instance de cette classe doit exister à un moment donné.
 *
 * Les instances de cette classe utilisent la valeur de
 * ArcaneMain::defaultApplicationInfo() pour s'initialiser et notamment
 * récupérer les arguments de la ligne de commande.
 *
 * Il faut appeler la méthode initialize() avant d'appeler d'autres
 * méthodes telles que createSubDomain(). Il est possible de modifier les
 * paramètres de création de l'application en modifiant les valeurs
 * de l'instance retournée par applicationBuildInfo().
 */
class ARCANE_IMPL_EXPORT ArcaneSimpleExecutor
{
  class Impl;
 public:

  ArcaneSimpleExecutor();
  ArcaneSimpleExecutor(const ArcaneSimpleExecutor&) = delete;
  ~ArcaneSimpleExecutor() noexcept(false);
  const ArcaneSimpleExecutor& operator=(const ArcaneSimpleExecutor&) = delete;

 public:

  ApplicationBuildInfo& applicationBuildInfo();
  const ApplicationBuildInfo& applicationBuildInfo() const;

  int initialize();
  ISubDomain* createSubDomain(const String& case_file_name);
  int runCode(IFunctor* f);

 private:

  Impl* m_p;

 private:

  void _checkInit();
  void _setDefaultVerbosityLevel(Integer level);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
