// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYUTILS_H
#define ARCANE_UTILS_MEMORYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arccore/common/MemoryAllocationArgs.h"
#include "arccore/common/MemoryUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MemoryUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur spécifique pour les accélérateurs.
 *
 * \deprecated Use MemoryUtils::getDefaultDataAllocator() instead.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: Use getDefaultDataAllocator() instead.")
ARCANE_UTILS_EXPORT IMemoryAllocator*
getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  //! Calcule une capacité adaptée pour une taille de \a size
  extern "C++" ARCANE_UTILS_EXPORT Int64
  computeCapacity(Int64 size);
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Redimensionne un tableau en ajoutant une réserve de mémoire.
 *
 * Le tableau \a array est redimensionné uniquement si \a new_size est
 * supérieure à la taille actuelle du tableau ou si \a force_resize est vrai.
 *
 * Si le tableau est redimensionné, on réserve une capacité supplémentaire
 * pour éviter de réallouer à chaque fois.
 *
 * \retval 2 si on a réalloué via reserve()
 * \retval 1 si on a re-dimensionné sans réallouer.
 * \retval 0 si aucune opération n'a eu lieu.
 */
template <typename DataType> inline Int32
checkResizeArrayWithCapacity(Array<DataType>& array, Int64 new_size, bool force_resize)
{
  Int32 ret_value = 0;
  Int64 s = array.largeSize();
  if (new_size > s || force_resize) {
    ret_value = 1;
    if (new_size > array.capacity()) {
      array.reserve(impl::computeCapacity(new_size));
      ret_value = 2;
    }
    array.resize(new_size);
  }
  return ret_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Redimensionne un tableau en ajoutant une réserve de mémoire.
 *
 * Cet appel est équivalent à checkResizeArrayWithCapacity(array, new_size, false).
 */
template <typename DataType> inline Int32
checkResizeArrayWithCapacity(Array<DataType>& array, Int64 new_size)
{
  return checkResizeArrayWithCapacity(array, new_size, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
