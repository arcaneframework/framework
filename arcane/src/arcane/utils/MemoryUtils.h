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

#include "arcane/utils/MemoryRessource.h"
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
 * \deprecated Use MemoryUtils::getDefaultDataAllocator() instead.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: Use getDefaultDataAllocator() instead.")
ARCANE_UTILS_EXPORT IMemoryAllocator*
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
copy(Span<DataType> destination, Span<const DataType> source,
     const RunQueue* queue = nullptr)
{
  ConstMemoryView input(asBytes(source));
  MutableMemoryView output(asWritableBytes(destination));
  copy(output, input, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copie de \a source vers \a destination en utilisant la file \a queue.
template <typename DataType> inline void
copy(SmallSpan<DataType> destination, SmallSpan<const DataType> source,
     const RunQueue* queue = nullptr)
{
  copy(Span<DataType>(destination), Span<const DataType>(source), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie sur l'hôte des données avec indirection.
 *
 * Copie dans \a destination les données de \a source
 * indexées par \a indexes
 *
 * L'opération est équivalente au pseudo-code suivant:
 *
 * \code
 * Int64 n = indexes.size();
 * for( Int64 i=0; i<n; ++i )
 *   destination[i] = source[indexes[i]];
 * \endcode
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre source.nbElement() >= indexes.size();
 */
extern "C++" ARCANE_UTILS_EXPORT void
copyToIndexesHost(MutableMemoryView destination, ConstMemoryView source,
                  Span<const Int32> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie sur l'hôte des données avec indirection.
 *
 * Copie dans \a destination les données de \a source
 * indexées par \a indexes
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destinationv[i] = source[indexes[i]];
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre source.nbElement() >= indexes.size();
 */
extern "C++" ARCANE_UTILS_EXPORT void
copyToIndexes(MutableMemoryView destination, ConstMemoryView source,
              SmallSpan<const Int32> indexes,
              RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie dans \a destination les données de \a source.
 *
 * Utilise std::memmove pour la copie.
 *
 * \pre source.bytes.size() >= destination.bytes.size()
 */
extern "C++" ARCANE_UTILS_EXPORT void
copyHost(MutableMemoryView destination, ConstMemoryView source);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie dans l'instance les données indexées de \a v.
 *
 * L'opération est équivalente au pseudo-code suivant:
 *
 * \code
 * Int64 n = indexes.size();
 * for( Int64 i=0; i<n; ++i )
 *   destination[indexes[i]] = source[i];
 * \endcode
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre destination.nbElement() >= indexes.size();
 */
extern "C++" ARCANE_UTILS_EXPORT void
copyFromIndexesHost(MutableMemoryView destination, ConstMemoryView source,
                    Span<const Int32> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie dans l'instance les données indexées de \a v.
 *
 * L'opération est équivalente au pseudo-code suivant:
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destination[indexes[i]] = source[i];
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre destination.nbElement() >= indexes.size();
 */
extern "C++" ARCANE_UTILS_EXPORT void
copyFromIndexes(MutableMemoryView destination, ConstMemoryView source,
                SmallSpan<const Int32> indexes, RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit une zone mémoire indexée avec une valeur.
 *
 * Remplit les indices \a indexes de la zone mémoire \a destination avec
 * la valeur de la zone mémoire \a source. \a source doit avoir une seule valeur.
 * La zone mémoire \a source être accessible depuis l'hôte.
 *
 * L'opération est équivalente au pseudo-code suivant:
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destination[indexes[i]] = source[0];
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre destination.nbElement() >= indexes.size();
 */
extern "C++" ARCANE_UTILS_EXPORT void
fillIndexes(MutableMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, const RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit une zone mémoire avec une valeur.
 *
 * Remplit les valeurs de la zone mémoire \a destination avec
 * la valeur de la zone mémoire \a source. \a source doit avoir une seule valeur.
 * La zone mémoire \a source être accessible depuis l'hôte.
 *
 * L'opération est équivalente au pseudo-code suivant:
 *
 * \code
 * Int32 n = nbElement();
 * for( Int32 i=0; i<n; ++i )
 *   destination[i] = source[0];
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 */
extern "C++" ARCANE_UTILS_EXPORT void
fill(MutableMemoryView destination, ConstMemoryView source,
     const RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
