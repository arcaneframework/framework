// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadBarrier.h                                            (C) 2000-2018 */
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

namespace Arccore
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

