// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProcessorAffinityService.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service de gestion de l'affinité des coeurs CPU.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
#define ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de gestion de l'affinité des coeurs CPU.
 */
class IProcessorAffinityService
{
 public:

  virtual ~IProcessorAffinityService() {} //<! Libère les ressources

 public:

  virtual void build() = 0;

 public:

  //! Affiche les informations complète de topologie via info()
  virtual void printInfos() = 0;

  /*!
   * \brief Retourne le cpuset pour le thread courant.
   *
   * La chaîne retournée est dans un format compatible avec celui
   * de taskset. Par exemple, on peut avoir des valeurs telles
   * que \a 'ff', '1, ou 'ffff1234,ff'.
   */
  virtual String cpuSetString() = 0;

  //! Contraint le thread courant à rester sur le coeur d'indice \a cpu
  virtual void bindThread(Int32 cpu) = 0;

  //! Nombre de coeurs CPU (-1 si inconnu)
  virtual Int32 numberOfCore() = 0;

  //! Nombre de sockets (-1 si inconnu)
  virtual Int32 numberOfSocket() = 0;

  //! Nombre de coeurs logiques (-1 si inconnu)
  virtual Int32 numberOfProcessingUnit() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
