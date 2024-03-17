// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.h                                               (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYUTILS_H
#define ARCANE_UTILS_MEMORYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryRessource.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/MemoryView.h"

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
 * \retval true si un redimensionnement a eu lieu
 * \retval false sinon
 */
template <typename DataType> inline bool
checkResizeArrayWithCapacity(Array<DataType>& array, Int64 new_size, bool force_resize)
{
  Int64 s = array.largeSize();
  if (new_size > s || force_resize) {
    if (new_size > array.capacity()) {
      array.reserve(impl::computeCapacity(new_size));
    }
    array.resize(new_size);
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copie de \a source vers \a destination en utilisant la file \a queue.
extern "C++" ARCANE_UTILS_EXPORT void
copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue = nullptr);

//! Copie de \a source vers \a destination en utilisant la file \a queue.
template <typename DataType> inline void
copy(Span<DataType> destination, Span<const DataType> source, const RunQueue* queue = nullptr)
{
  ConstMemoryView input(asBytes(source));
  MutableMemoryView output(asWritableBytes(destination));
  copy(output, input, queue);
}

//! Copie de \a source vers \a destination en utilisant la file \a queue.
template <typename DataType> inline void
copy(SmallSpan<DataType> destination, SmallSpan<const DataType> source, const RunQueue* queue = nullptr)
{
  copy(Span<DataType>(destination), Span<const DataType>(source), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
