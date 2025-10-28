// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtilsInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire internes à Arcane.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_MEMORYUTILSINTERNAL_H
#define ARCANE_UTILS_INTERNAL_MEMORYUTILSINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MemoryUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le gestionnaire de ressource mémoire pour les données.
 *
 * Le gestionnaire doit rester valide durant toute l'exécution du programme.
 *
 * Retourne l'ancien gestionnaire.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng*
setDataMemoryResourceMng(IMemoryRessourceMng* mng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de ressource mémoire pour les données.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng*
getDataMemoryResourceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne l'allocateur spécifique pour les accélérateurs.
 *
 * Retourne l'ancien allocateur utilisé. L'allocateur spécifié doit rester
 * valide durant toute la durée de vie de l'application.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
setAcceleratorHostMemoryAllocator(IMemoryAllocator* a);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne la ressource mémoire utilisée pour l'allocateur
 * mémoire des données.
 *
 * \sa getDefaultDataMemoryResource();
 */
extern "C++" ARCANE_UTILS_EXPORT void
setDefaultDataMemoryResource(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
