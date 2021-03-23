// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectExecutionContext.h                                   (C) 2000-2021 */
/*                                                                           */
/* Implémentation de la classe de gestion de l'exécution.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_IDIRECTEXECUTIONCONTEXT_H
#define ARCANE_LAUNCHER_IDIRECTEXECUTIONCONTEXT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation de la classe de gestion de l'exécution.
 */
class ARCANE_LAUNCHER_EXPORT IDirectExecutionContext
{
 public:
  virtual ~IDirectExecutionContext() = default;
 public:
  /*!
   * \brief Créé un sous-domaine en séquentiel sans jeu de données
   */
  virtual ISubDomain* createSequentialSubDomain() =0;

  /*!
   * \brief Créé un sous-domaine en séquentiel avec le fichier de
   * jeu de données ayant pour nom \a case_file_name.
   */
  virtual ISubDomain* createSequentialSubDomain(const String& case_file_name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
