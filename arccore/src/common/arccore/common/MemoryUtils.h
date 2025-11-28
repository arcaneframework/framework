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
#ifndef ARCCORE_COMMON_MEMORYUTILS_H
#define ARCCORE_COMMON_MEMORYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

#include "arccore/base/MemoryView.h"

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
extern "C++" ARCCORE_DEPRECATED_REASON("Y2024: Use getDefaultDataAllocator() instead.")
ARCCORE_COMMON_EXPORT IMemoryAllocator*
getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ressource mémoire utilisée par l'allocateur par défaut pour les données.
 *
 * Par défaut, si un runtime accélérateur est initialisé, la ressource
 * associée est eMemoryResource::UnifiedMemory. Sinon, il s'agit de
 * eMemoryResource::Host.
 *
 * \sa getDefaultDataAllocator();
 */
extern "C++" ARCCORE_COMMON_EXPORT eMemoryResource
getDefaultDataMemoryResource();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la ressource mémoire par son nom.
 *
 * Le nom correspond au nom de la valeur de l'énumération (par exemple
 * 'Device' pour eMemoryResource::Device.
 *
 * Si \a name est nul, retourne eMemoryResource::Unknown.
 * Si \a name ne correspondant pas à une valeur valide, lève une exception.
 */
extern "C++" ARCCORE_COMMON_EXPORT eMemoryResource
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
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
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
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
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
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
getDeviceOrHostAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données essentiellement en
 * lecture.
 *
 * Cet appel est équivalent à getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead).
 */
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
getAllocatorForMostlyReadOnlyData();

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
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
getAllocationOptions(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour la ressource \a mem_resource.
 *
 * \sa getAllocationOptions().
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
getAllocator(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie de \a source vers \a destination en utilisant la file \a queue.
 *
 * Il est possible de spécifier la ressource mémoire où se trouve la source
 * et la destination. Si on ne les connait pas, il est préférable d'utiliser
 * la surcharge copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue).
 */
extern "C++" ARCCORE_COMMON_EXPORT void
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
extern "C++" ARCCORE_COMMON_EXPORT void
copyHostWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
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
 *   destination[i] = source[indexes[i]];
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre source.nbElement() >= indexes.size();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
copyWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
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
extern "C++" ARCCORE_COMMON_EXPORT void
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
extern "C++" ARCCORE_COMMON_EXPORT void
copyHostWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
                               Span<const Int32> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie mémoire avec indirection
 *
 * Copie les données de \a source dans \a destination pour les indices
 * spécifiés par \a indexes.
 *
 * L'opération est équivalente au pseudo-code suivant :
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
extern "C++" ARCCORE_COMMON_EXPORT void
copyWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
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
extern "C++" ARCCORE_COMMON_EXPORT void
fillIndexed(MutableMemoryView destination, ConstMemoryView source,
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
 * L'opération est équivalente au pseudo-code suivant :
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
extern "C++" ARCCORE_COMMON_EXPORT void
fill(MutableMemoryView destination, ConstMemoryView source,
     const RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie dans \a destination les données de \a source indexées.
 *
 * L'opération est équivalente au pseudo-code suivant :
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i ){
 *   Int32 index0 = indexes[ (i*2)   ];
 *   Int32 index1 = indexes[ (i*2)+1 ];
 *   destination[i] = source[index0][index1];
 * }
 * \endcode
 *
 * Le tableau \a indexes doit avoir une taille multiple de 2. Les valeurs
 * paires servent à indexer le premier tableau et les valeurs impaires le 2ème.
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre destination.nbElement() >= indexes.size();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
copyWithIndexedSource(MutableMemoryView destination, ConstMultiMemoryView source,
                      SmallSpan<const Int32> indexes, RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie les éléments indéxés de \a destination avec les données de \a source.
 *
 * L'opération est équivalente au pseudo-code suivant :
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i ){
 *   Int32 index0 = indexes[ (i*2)   ];
 *   Int32 index1 = indexes[ (i*2)+1 ];
 *   destination[index0][index1] = source[i];
 * }
 * \endcode
 *
 * Le tableau \a indexes doit avoir une taille multiple de 2. Les valeurs
 * paires servent à indexer le premier tableau et les valeurs impaires le 2ème.
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == v.datatypeSize();
 * \pre source.nbElement() >= indexes.size();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
copyWithIndexedDestination(MutableMultiMemoryView destination, ConstMemoryView source,
                           SmallSpan<const Int32> indexes, RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les éléments indéxés de \a destination avec la donnée \a source.
 *
 * \a source doit avoir une seule valeur. Cette valeur sera utilisée
 * pour remplir les valeurs de l'instance aux indices spécifiés par
 * \a indexes. Elle doit être accessible depuis l'hôte.
 *
 * L'opération est équivalente au pseudo-code suivant :
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i ){
 *   Int32 index0 = indexes[ (i*2)   ];
 *   Int32 index1 = indexes[ (i*2)+1 ];
 *   destination[index0][index1] = source[0];
 * }
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 * \pre destination.nbElement() >= indexes.size();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
fillIndexed(MutableMultiMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les éléments de \a destination avec la valeur \a source.
 *
 * \a source doit avoir une seule valeur. Elle doit être accessible depuis l'hôte.
 *
 * L'opération est équivalente au pseudo-code suivant :
 *
 * \code
 * Int32 n = nbElement();
 * for( Int32 i=0; i<n; ++i ){
 *   Int32 index0 = (i*2);
 *   Int32 index1 = (i*2)+1;
 *   destination[index0][index1] = source[0];
 * }
 * \endcode
 *
 * Si \a run_queue n'est pas nul, elle sera utilisée pour la copie.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
fill(MutableMultiMemoryView destination, ConstMemoryView source,
     RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
