// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAsyncQueue.h                                               (C) 2000-2019 */
/*                                                                           */
/* File asynchrone permettant d'échanger des informations entre threads.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_IASYNCQUEUE_H
#define ARCANE_PARALLEL_THREAD_IASYNCQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief File asynchrone permettant d'échanger des informations entre threads
 */
class IAsyncQueue
{
 public:
  virtual ~IAsyncQueue() = default;
 public:
  //! Ajoute \a v dans la file.
  virtual void push(void* v) =0;
  /*!
   * \brief Récupère la première valeur de la file et bloque s'il n'y en a pas.
   */
  virtual void* pop() =0;
  /*!
   * \brief Récupère la première valeur s'il y en. Retourne `nullptr` sinon.
   */
  virtual void* tryPop() =0;
 public:
  static IAsyncQueue* createQueue();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

