﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueStream.h                                           (C) 2000-2024 */
/*                                                                           */
/* Interface d'un flux d'exécution pour une RunQueue.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IRUNQUEUESTREAM_H
#define ARCANE_ACCELERATOR_IRUNQUEUESTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un flux d'exécution pour une RunQueue.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueStream
{
 public:

  virtual ~IRunQueueStream() = default;

 public:

  //! Notification avant le lancement de la commande
  virtual void notifyBeginLaunchKernel(impl::RunCommandImpl& command) = 0;

  /*!
   * \brief Notification de fin de lancement de la commande.
   *
   * En mode asynchrone, la commande peut continuer à s'exécuter en tâche de fond.
   */
  virtual void notifyEndLaunchKernel(impl::RunCommandImpl& command) = 0;

  /*!
   * \brief Bloque jusqu'à ce que toutes les actions associées à cette file
   * soient terminées.
   *
   * Cela comprend les commandes (RunCommandImpl) et les autres actions telles
   * que les copies mémoire asynchrones.
   */
  virtual void barrier() = 0;

  //! Effectue une copie entre deux zones mémoire
  virtual void copyMemory(const MemoryCopyArgs& args) = 0;

  //! Effectue un pré-chargement d'une zone mémoire
  virtual void prefetchMemory(const MemoryPrefetchArgs& args) = 0;

 public:

  //! Pointeur sur la structure interne dépendante de l'implémentation
  virtual void* _internalImpl() = 0;

  //! Barrière sans exception. Retourne \a true en cas d'erreur
  virtual bool _barrierNoException() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
