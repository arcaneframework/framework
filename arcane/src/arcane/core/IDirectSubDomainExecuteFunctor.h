// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectSubDomainExecuteFunctor.h                            (C) 2000-2021 */
/*                                                                           */
/* Interface d'un fonctor d'exécution directe avec sous-domaine.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDIRECTSUBDOMAINEXECUTEFUNCTOR_H
#define ARCANE_IDIRECTSUBDOMAINEXECUTEFUNCTOR_H
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
 * \brief Interface d'un fonctor pour exécuter du code directement après
 * la création d'un sous-domaine sans passer par la boucle en temps.
 */
class ARCANE_CORE_EXPORT IDirectSubDomainExecuteFunctor
{
 public:

  virtual ~IDirectSubDomainExecuteFunctor() = default;

 public:

  //! Exécute l'opération du fonctor
  virtual int execute() =0;

  /*!
   * \brief Positionne le sous-domaine associé.
   * Cette méthode doit être appelée avant execute()
   */
  virtual void setSubDomain(ISubDomain* sd) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
