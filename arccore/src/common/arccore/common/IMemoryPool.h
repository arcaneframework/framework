// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryPool.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un pool mémoire.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYPOOL_H
#define ARCCORE_COMMON_IMEMORYPOOL_H
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
 * \brief Interface d'un pool mémoire.
 */
class ARCCORE_COMMON_EXPORT IMemoryPool
{
 public:

  virtual ~IMemoryPool() = default;

 public:

  /*!
   * \brief Positionne la taille en octet à partir de laquelle
   * on ne conserve pas un bloc dans le cache.
   *
   * Cette méthode ne peut être appelée que s'il n'y a aucun bloc dans le
   * cache.
   */
  virtual void setMaxCachedBlockSize(Int32 v) = 0;

  //! Libère la mémoire dans le cache
  virtual void freeCachedMemory() = 0;

  //! Taille totale (en octet) allouée dans le pool mémoire
  virtual size_t totalAllocated() const = 0;

  //! Taille totale (en octet) dans le cache
  virtual size_t totalCached() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
