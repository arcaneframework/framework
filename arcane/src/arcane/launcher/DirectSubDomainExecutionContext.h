// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectSubDomainExecutionContext.h                           (C) 2000-2021 */
/*                                                                           */
/* Contexte d'exécution directe avec création d'un sous-domaine.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_DIRECTSUBDOMAINEXECUTIONCONTEXT_H
#define ARCANE_LAUNCHER_DIRECTSUBDOMAINEXECUTIONCONTEXT_H
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
 * \brief Contexte d'exécution directe avec création d'un sous-domaine.
 */
class ARCANE_LAUNCHER_EXPORT DirectSubDomainExecutionContext
{
  class Impl;
  friend class ArcaneLauncherDirectExecuteFunctor;

 protected:

  // Méthode protégée pour qu'on ne puisse créer une instance que via Arcane
  DirectSubDomainExecutionContext(ISubDomain* sd);

 public:

  ~DirectSubDomainExecutionContext();
  DirectSubDomainExecutionContext(const DirectSubDomainExecutionContext&) = delete;
  DirectSubDomainExecutionContext& operator=(const DirectSubDomainExecutionContext&) = delete;

 public:

  //! Sous domaine
  ISubDomain* subDomain() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
