// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadBarrier.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface d'une barrière avec les threads.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_ITHREADBARRIER_H
#define ARCCORE_CONCURRENCY_ITHREADBARRIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une barrière entre threads.
 *
 * Une fois créée (via IThreadImplementation::createBarrier()),
 * la barrière doit être initialisée
 * via init() pour \a n threads. Ensuite, chaque thread doit
 * appeler la méthode wait() pour attendre que tous les
 * autres thread arrivent à ce même point.
 * La barrière peut être utilisée plusieurs fois.
 * Pour détruire la barrière, il faut appeler destroy(). Cela libère aussi
 * l'instance qui ne doit ensuite plus être utilisée.
 */
class ARCCORE_CONCURRENCY_EXPORT IThreadBarrier
{
 protected:

  virtual ~IThreadBarrier(){}

 public:

  //! Initialise la barrière pour \a nb_thread.
  virtual void init(Integer nb_thread) =0;

  //! Détruit la barrière.
  virtual void destroy() =0;

  /*!
   * \brief Bloque et attend que tous les threads appellent cette méthode.
   *
   * \retval true si on est le dernier thread qui appelle cette méthode.
   * \retval false sinon.
   */
  virtual bool wait() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

