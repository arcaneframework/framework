// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectExecutionContext.h                                    (C) 2000-2021 */
/*                                                                           */
/* Contexte d'exécution directe.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_DIRECTEXECUTIONCONTEXT_H
#define ARCANE_LAUNCHER_DIRECTEXECUTIONCONTEXT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;
class IDirectExecutionContext;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution directe.
 */
class ARCANE_LAUNCHER_EXPORT DirectExecutionContext
{
 public:
  DirectExecutionContext(IDirectExecutionContext* ctx);
  DirectExecutionContext(const DirectExecutionContext&) = delete;
  DirectExecutionContext& operator=(const DirectExecutionContext&) = delete;
 public:
  /*!
   * \brief Créé un sous-domaine en séquentiel sans jeu de données
   */
  ISubDomain* createSequentialSubDomain();
  /*!
   * \brief Créé un sous-domaine en séquentiel avec le fichier de
   * jeu de données ayant pour nom \a case_file_name.
   */
  ISubDomain* createSequentialSubDomain(const String& case_file_name);
 private:
  IDirectExecutionContext* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
