// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectExecution.h                                          (C) 2000-2021 */
/*                                                                           */
/* Interface d'un service d'exécution direct.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDIRECTEXECUTION_H
#define ARCANE_IDIRECTEXECUTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service d'exécution direct.
 *
 * Un service d'exécution direct est un service qui exécute une opération
 * unique en remplacement de la boucle en temps, en général pour effectuer
 * des tests internes à Arcane.
 *
 * Une fois l'opération terminée, le code s'arrête.
 *
 * Ce service peut être associé à une application et dans ce
 * case il n'a pas de sous-domaine ou de maillage et il faut positionner
 * le gestionnaire de parallèlisme avant l'exécution.
 */
class ARCANE_CORE_EXPORT IDirectExecution
{
 public:

  virtual ~IDirectExecution() {} //!< Libère les ressources.

 public:

  virtual void build() =0;

 public:

  //! Exécute l'opération du service
  virtual void execute() =0;

  //! Vrai si le service est actif
  virtual bool isActive() const =0;

  /*!
   * \internal.
   * \brief Positionne le gestionnaire de parallèlisme associé.
   * Cette méthode doit être appelée avant execute()
   */
  virtual void setParallelMng(IParallelMng* pm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
