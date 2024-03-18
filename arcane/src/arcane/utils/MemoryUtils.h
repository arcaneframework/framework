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
 * Si un runtime accélérateur est initialisé, l'allocateur retourné permet
 * d'allouer en mémoire unifiée et donc la zone allouée sera accessible à la
 * fois sur l'accélérateur et sur l'hôte. Sinon, retourne un allocateur
 * aligné.
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
 * Cette fonction retourne l'allocateur de getDefaulDataAllocator() mais
 * ajoute les informations de gestion mémoire spécifiées par \a hint.
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getDefaultDataAllocator(eMemoryLocationHint hint);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données essentiellement en
 * lecture.
 *
 * Cet appel est équivalent à getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead).
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getAllocatorForMostlyReadOnlyData();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour la ressource \a mem_ressource.
 *
 * Lève une exception si aucune allocateur n'est disponible pour la ressource
 * (par exemple si on demande eMemoryRessource::Device et qu'il n'y a pas de
 * support pour les accélérateurs.
 *
 * La ressource eMemoryRessource::UnifiedMemory est toujours disponible. Si
 * aucun runtime accélérateur n'est chargé, alors c'est équivalent à
 * eMemoryRessource::Host.
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getAllocationOptions(eMemoryRessource mem_ressource);

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
/*!
 * \brief Copie de \a source vers \a destination en utilisant la file \a queue.
 *
 * Il est possible de spécifier la ressource mémoire où se trouve la source
 * et la destination. Si on ne les connait pas, il est préférable d'utiliser
 * la surcharge copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue).
 */
extern "C++" ARCANE_UTILS_EXPORT void
copy(MutableMemoryView destination, eMemoryRessource destination_mem,
     ConstMemoryView source, eMemoryRessource source_mem,
     const RunQueue* queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copie de \a source vers \a destination en utilisant la file \a queue.
inline void
copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue = nullptr)
{
  eMemoryRessource mem_type = eMemoryRessource::Unknown;
  copy(destination, mem_type, source, mem_type, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
