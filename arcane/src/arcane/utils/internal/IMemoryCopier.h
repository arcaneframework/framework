// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryCopier.h                                             (C) 2000-2024 */
/*                                                                           */
/* Interface pour les copies mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_IMEMORYCOPIER_H
#define ARCANE_UTILS_INTERNAL_IMEMORYCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMemoryRessourceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour les copies mémoire avec support des accélérateurs.
 */
class ARCANE_UTILS_EXPORT IMemoryCopier
{
 public:

  virtual ~IMemoryCopier() = default;

 public:

  /*!
   * \brief Copie les données de \a from vers \a to avec la queue \a queue.
   *
   * \a queue peut-être nul.
   */
  virtual void copy(ConstMemoryView from, eMemoryRessource from_mem,
                    MutableMemoryView to, eMemoryRessource to_mem, RunQueue* queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
