﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryRessourceMngInternal.h                               (C) 2000-2022 */
/*                                                                           */
/* Partie interne à Arcane de 'IMemoryRessourceMng'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_IMEMORYRESSOURCEMNGINTERNAL_H
#define ARCANE_UTILS_INTERNAL_IMEMORYRESSOURCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMemoryRessourceMng.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT IMemoryCopier
{
 public:

  //! Copie les données de \a from vers \a to
  virtual void copy(Span<const std::byte> from, eMemoryRessource from_mem,
                    Span<std::byte> to, eMemoryRessource to_mem) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne à Arcane de 'IMemoryRessourceMng'.
 */
class ARCANE_UTILS_EXPORT IMemoryRessourceMngInternal
{
 public:

  virtual ~IMemoryRessourceMngInternal() = default;

  virtual void copy(Span<const std::byte> from, eMemoryRessource from_mem,
                    Span<std::byte> to, eMemoryRessource to_mem) = 0;

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
