// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryRessource.h                                           (C) 2000-2022 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYRESSOURCE_H
#define ARCANE_UTILS_MEMORYRESSOURCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste des ressources mémoire disponibles
enum class eMemoryRessource
{
  Unknown = 0,
  Host,
  Accelerator,
  UnifiedMemory
};

//! Nombre de valeurs valides pour eMemoryRessource
static constexpr int NB_MEMORY_RESSOURCE = 4;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o,eMemoryRessource r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
