// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryRessourceMngInternal.h                               (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de 'IMemoryRessourceMng'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_IMEMORYRESSOURCEMNGINTERNAL_H
#define ARCANE_UTILS_INTERNAL_IMEMORYRESSOURCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/IMemoryCopier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne à Arcane de 'IMemoryRessourceMng'.
 */
class ARCANE_UTILS_EXPORT IMemoryRessourceMngInternal
{
 public:

  virtual ~IMemoryRessourceMngInternal() = default;

 public:

  virtual void copy(ConstMemoryView from, eMemoryRessource from_mem,
                    MutableMemoryView to, eMemoryRessource to_mem, const RunQueue* queue) = 0;

 public:

  virtual void setAllocator(eMemoryRessource r, IMemoryAllocator* allocator) = 0;

  virtual void setCopier(IMemoryCopier* copier) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
