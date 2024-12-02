﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
 * \brief Ressource mémoire utilisée par l'allocateur par défaut pour les données.
 *
 * Par défaut, si un runtime accélérateur est initialisé, la ressource
 * associé est eMemoryResource::UnifiedMemory. Sinon, il s'agit de
 * eMemoryResource::Host.
 *
 * \sa getDefaultDataAllocator();
 */
extern "C++" ARCANE_UTILS_EXPORT eMemoryResource
getDefaultDataMemoryResource();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la ressource mémoire par son nom.
 *
 * Le nom correspond au nom de la valeur de l'énumération (par exemple
 * 'Device' pour eMemoryResource::Device.
 *
 * Si \a name est nul, retourn eMemoryResource::Unknown.
 * Si \a name ne correspondant pas à une valeur valide, lève une exception.
 */
extern "C++" ARCANE_UTILS_EXPORT eMemoryResource
getMemoryResourceFromName(const String& name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données.
 *
 * L'allocateur par défaut pour les données est un allocateur qui permet
 * d'accéder à la zone mémoire à la fois par l'hôte et l'accélérateur.
 *
 * Il est possible de récupérer la ressource mémoire associée via
 * getDefaultDataMemoryResource();
 *
 * Cet appel est équivalent à getAllocator(getDefaultDataMemoryResource()).
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
 * \brief Retourne l'allocateur sur l'hôte ou sur le device.
 *
 * Si un runtime accélérateur est initialisé, l'allocateur retourné permet
 * d'allouer en utilisant la mémoire de l'accélérateur par défaut
 * (eMemoryResource::Device). Sinon, utilise l'allocateur de l'hôte
 * (eMemoryResource::Host).
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getDeviceOrHostAllocator();

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
 * \brief Allocateur spécifique pour les accélérateurs.
 *
 * Si non nul, cet allocateur permet d'allouer de la mémoire sur l'hôte en
 * utilisant le runtime spécique de l'allocateur.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocation par défaut pour la ressource \a mem_resource.
 *
 * Lève une exception si aucune allocateur n'est disponible pour la ressource
 * (par exemple si on demande eMemoryResource::Device et qu'il n'y a pas de
 * support pour les accélérateurs.
 *
 * La ressource eMemoryResource::UnifiedMemory est toujours disponible. Si
 * aucun runtime accélérateur n'est chargé, alors c'est équivalent à
 * eMemoryResource::Host.
 */
extern "C++" ARCANE_UTILS_EXPORT MemoryAllocationOptions
getAllocationOptions(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour la ressource \a mem_resource.
 *
 * \sa getAllocationOptions().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getAllocator(eMemoryResource mem_resource);

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
 * \brief Copie de \a source vers \a destination en utilisant la file \a queue.
 *
 * Il est possible de spécifier la ressource mémoire où se trouve la source
 * et la destination. Si on ne les connait pas, il est préférable d'utiliser
 * la surcharge copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue).
 */
extern "C++" ARCANE_UTILS_EXPORT void
copy(MutableMemoryView destination, eMemoryResource destination_mem,
     ConstMemoryView source, eMemoryResource source_mem,
     const RunQueue* queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copie de \a source vers \a destination en utilisant la file \a queue.
inline void
copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue = nullptr)
{
  eMemoryResource mem_type = eMemoryResource::Unknown;
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
