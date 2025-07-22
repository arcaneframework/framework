// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMngInternal.h                                      (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de IParallelMng.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IPARALLELMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_IPARALLELMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MessagePassing
{
  class IMachineMemoryWindowBase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IParallelMng.
 */
class ARCANE_CORE_EXPORT IParallelMngInternal
{
 public:

  virtual ~IParallelMngInternal() = default;

 public:

  //! Runner par défaut. Peut être nul
  virtual Runner runner() const = 0;

  //! File par défaut pour les messages. Peut être nul
  virtual RunQueue queue() const = 0;

  /*!
   * \brief Indique si l'implémentation gère les accélérateurs.
   *
   * Si c'est le cas on peut utiliser directement la mémoire de l'accélérateur
   * dans les appels MPI ce qui permet d'éviter d'éventuelles recopies.
   */
  virtual bool isAcceleratorAware() const = 0;

  //! Créé un sous IParallelMng de manière similaire à MPI_Comm_split.
  virtual Ref<IParallelMng> createSubParallelMngRef(Int32 color, Int32 key) = 0;

  virtual void setDefaultRunner(const Runner& runner) = 0;

  /*!
   * \brief Méthode permettant de créer une fenêtre mémoire sur le noeud.
   *
   * Appel collectif.
   *
   * \param sizeof_segment La taille de notre segment (en octet).
   * \param sizeof_type La taille d'un élément du segment (en octet).
   * \return Une référence vers la nouvelle fenêtre.
   */
  virtual Ref<MessagePassing::IMachineMemoryWindowBase> createMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
