// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.h                                               (C) 2000-2023 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYUTILS_H
#define ARCANE_UTILS_MEMORYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryRessource.h"
#include "arcane/utils/UtilsTypes.h"

#include "arccore/collections/MemoryAllocationArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MemoryUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données.
 *
 * Cette allocateur utilise celui getAcceleratorHostMemoryAllocator()
 * s'il est disponible, sinon il utilise un allocateur aligné.
 *
 * Il est garanti que l'allocateur retourné permettra d'utiliser la donnée
 * à la fois sur accélerateur et sur l'hôte si cela est disponible.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getDefaultDataAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données avec informations sur
 * la localisation attendue.
 *
 * \sa getDefaultDataAllocator()
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getDefaultDataAllocator(eMemoryLocationHint hint);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données essentiellements en
 * lecture.
 *
 * Cet appel est équivalent à getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead).
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getAllocatorForMostlyReadOnlyData();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
