// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryResourceMngInternal.h                                (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de 'IMemoryResourceMng'.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_IMEMORYRESOURCEMNGINTERNAL_H
#define ARCCORE_COMMON_INTERNAL_IMEMORYRESOURCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/IMemoryCopier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne à Arcane de 'IMemoryRessourceMng'.
 */
class ARCCORE_COMMON_EXPORT IMemoryResourceMngInternal
{
 public:

  virtual ~IMemoryResourceMngInternal() = default;

 public:

  virtual void copy(ConstMemoryView from, eMemoryResource from_mem,
                    MutableMemoryView to, eMemoryResource to_mem, const RunQueue* queue) = 0;

 public:

  //! Positionne l'allocateur pour la ressource \a r
  virtual void setAllocator(eMemoryResource r, IMemoryAllocator* allocator) = 0;

  //! Positionne l'instance gérant les copies.
  virtual void setCopier(IMemoryCopier* copier) = 0;

  //! Indique si un accélérateur est disponible.
  virtual void setIsAccelerator(bool v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
