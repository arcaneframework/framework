// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryResourceMng.h                                        (C) 2000-2025 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYRESOURCEMNG_H
#define ARCCORE_COMMON_IMEMORYRESOURCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des ressources mémoire pour les CPU et accélérateurs.
 */
class ARCCORE_COMMON_EXPORT IMemoryResourceMng
{
 public:

  virtual ~IMemoryResourceMng() = default;

 public:

  /*!
   * \brief Allocateur mémoire pour la ressource \a r.
   *
   * Lève une exception si aucun allocateur pour la ressource \a v existe.
   */
  virtual IMemoryAllocator* getAllocator(eMemoryResource r) = 0;

  /*!
   * \brief Allocateur mémoire pour la ressource \a r.
   *
   * Si aucun allocateur pour la ressoruce \a v existe, lève une
   * exception si \a throw_if_not_found est vrai ou retourne \a nullptr
   * si \a throw_if_not_found est faux.
   */
  virtual IMemoryAllocator* getAllocator(eMemoryResource r, bool throw_if_not_found) = 0;

 public:

  //! Interface interne
  virtual IMemoryResourceMngInternal* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
