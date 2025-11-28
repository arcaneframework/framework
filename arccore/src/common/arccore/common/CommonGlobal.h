// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonGlobal.h                                              (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Common' de 'Arccore'.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_COMMONGLOBAL_H
#define ARCCORE_COMMON_COMMONGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_common)
#define ARCCORE_COMMON_EXPORT ARCCORE_EXPORT
#define ARCCORE_COMMON_EXTERN_TPL
#else
#define ARCCORE_COMMON_EXPORT ARCCORE_IMPORT
#define ARCCORE_COMMON_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class RunQueue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour l'instant on doit laisser ArrayTraits dans le namespace Arccore
// pour des raisons de compatibilité avec la macro ARCCORE_DEFINE_ARRAY_PODTYPE
namespace Arccore
{
template <typename DataType> class ArrayTraits;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arcane::Accelerator::RunQueue;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMemoryResourceMngInternal;
class IMemoryResourceMng;
class IMemoryCopier;

class IMemoryAllocator;
class AllocatedMemoryInfo;
class ArrayDebugInfo;
class MemoryAllocationArgs;
class MemoryAllocationOptions;
class PrintableMemoryAllocator;
class AlignedMemoryAllocator;
class DefaultMemoryAllocator;

class ArrayImplBase;
class ArrayMetaData;
template <typename DataType> class ArrayImplT;
template <typename DataType> class Array;
template <typename DataType> class AbstractArray;
template <typename DataType> class UniqueArray;
template <typename DataType> class SharedArray;
using Arccore::ArrayTraits;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indices sur la localisation mémoire attendue
enum class eMemoryLocationHint : int8_t
{
  //! Aucune indice
  None = 0,
  //! Indique que la donnée sera plutôt utilisée sur accélérateur
  MainlyDevice = 1,
  //! Indique que la donnée sera plutôt utilisée sur CPU
  MainlyHost = 2,
  /*!
   * \brief Indique que la donnée sera utilisée à la fois sur accélérateur et
   * sur CPU et qu'elle ne sera pas souvent modifiée.
   */
  HostAndDeviceMostlyRead = 3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Localisation physique d'une adresse mémoire.
 *
 * Pour les valeurs ManagedMemoryDevice et ManagedMemoryHost il s'agit d'une
 * indication car il n'y a pas de moyen simple de savoir où se trouve
 * réellement la mémoire.
 */
enum class eHostDeviceMemoryLocation : int8_t
{
  //! Localisation inconnue
  Unknown = 0,
  //! La mémoire est sur accélérateur
  Device = 1,
  //! La mémoire est sur l'hôte.
  Host = 2,
  //! La mémoire est de la mémoire managée sur accélérateur
  ManagedMemoryDevice = 3,
  //! La mémoire est de la mémoire managée sur l'hôte.
  ManagedMemoryHost = 4,
};

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eHostDeviceMemoryLocation r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste des ressources mémoire disponibles.
 */
enum class eMemoryResource
{
  //! Valeur inconnue ou non initialisée
  Unknown = 0,
  //! Alloue sur l'hôte.
  Host,
  //! Alloue sur l'hôte.
  HostPinned,
  //! Alloue sur le device
  Device,
  //! Alloue en utilisant la mémoire unifiée.
  UnifiedMemory
};

//! Nombre de valeurs valides pour eMemoryResource
static constexpr int ARCCORE_NB_MEMORY_RESOURCE = 5;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryResource r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Taille du padding pour les index dans les opérations SIMD.
 *
 * Afin d'avoir le même code quel que soit le mécanisme de vectorisation
 * utilisé, cette valeur est fixe et correspond au plus grand vecteur SIMD.
 *
 * \sa arcanedoc_simd
 */
static const Integer SIMD_PADDING_SIZE = 8;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

