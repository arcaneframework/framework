// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryCopier.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface pour les copies mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_IMEMORYCOPIER_H
#define ARCCORE_COMMON_INTERNAL_IMEMORYCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryResourceMng.h"
#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour les copies mémoire avec support des accélérateurs.
 */
class ARCCORE_COMMON_EXPORT IMemoryCopier
{
 public:

  virtual ~IMemoryCopier() = default;

 public:

  /*!
   * \brief Copie les données de \a from vers \a to avec la queue \a queue.
   *
   * \a queue peut-être nul.
   */
  virtual void copy(ConstMemoryView from, eMemoryResource from_mem,
                    MutableMemoryView to, eMemoryResource to_mem,
                    const RunQueue* queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
