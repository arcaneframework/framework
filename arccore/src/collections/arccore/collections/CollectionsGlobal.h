// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CollectionsGlobal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Collections' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_COLLECTIONSGLOBAL_H
#define ARCCORE_COLLECTIONS_COLLECTIONSGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"
#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_collections)
#define ARCCORE_COLLECTIONS_EXPORT ARCCORE_EXPORT
#define ARCCORE_COLLECTIONS_EXTERN_TPL
#else
#define ARCCORE_COLLECTIONS_EXPORT ARCCORE_IMPORT
#define ARCCORE_COLLECTIONS_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour l'instant on doit laisser ArrayTraits dans le namespace Arccore
// pour des raisons de compatibilité avec la macro ARCCORE_DEFINE_ARRAY_PODTYPE
namespace Arccore
{
template <typename DataType> class ArrayTraits;
}

namespace Arcane
{
class IMemoryAllocator;
class PrintableMemoryAllocator;
class AlignedMemoryAllocator;
class DefaultMemoryAllocator;
class ArrayImplBase;
class ArrayMetaData;
class MemoryAllocationArgs;
class MemoryAllocationOptions;
class ArrayDebugInfo;
template <typename DataType> class ArrayImplT;
template <typename DataType> class Array;
template <typename DataType> class AbstractArray;
template <typename DataType> class UniqueArray;
template <typename DataType> class SharedArray;
template <typename DataType> class Array2;
template <typename DataType> class UniqueArray2;
template <typename DataType> class SharedArray2;
using Arccore::ArrayTraits;

// Les trois types suivants sont conservés pour compatibilité avec l'existant.
// Il faudra les rendre optionnels mi-2025 et les supprimer en 2026
using IMemoryAllocator3 = IMemoryAllocator;
using AlignedMemoryAllocator3 = AlignedMemoryAllocator;
using DefaultMemoryAllocator3 = DefaultMemoryAllocator;
} // namespace Arcane

namespace Arcane::Accelerator
{
class RunQueue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arcane::Accelerator::RunQueue;

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
extern "C++" ARCCORE_COLLECTIONS_EXPORT std::ostream&
operator<<(std::ostream& o, eHostDeviceMemoryLocation r);


//! Typedef pour la version Arcane historique (avec 2's')
// TODO: A rendre obsolète (juin 2025)
using eMemoryRessource = Arcane::eMemoryResource;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une zone mémoire allouée.
 */
class AllocatedMemoryInfo
{
 public:

  AllocatedMemoryInfo() = default;
  explicit AllocatedMemoryInfo(void* base_address)
  : m_base_address(base_address)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(size)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size, Int64 capacity)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(capacity)
  {}

  //! Adresse du début de la zone allouée.
  void* baseAddress() const { return m_base_address; }
  //! Taille en octets de la zone mémoire utilisée. (-1) si inconnue
  Int64 size() const { return m_size; }
  //! Taille en octets de la zone mémoire allouée. (-1) si inconnue
  Int64 capacity() const { return m_capacity; }

 public:

  void* m_base_address = nullptr;
  Int64 m_size = -1;
  Int64 m_capacity = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::AbstractArray;
using Arcane::AlignedMemoryAllocator;
using Arcane::AlignedMemoryAllocator3;
using Arcane::AllocatedMemoryInfo;
using Arcane::ARCCORE_NB_MEMORY_RESOURCE;
using Arcane::Array;
using Arcane::Array2;
using Arcane::ArrayMetaData;
using Arcane::DefaultMemoryAllocator;
using Arcane::DefaultMemoryAllocator3;
using Arcane::eHostDeviceMemoryLocation;
using Arcane::eMemoryLocationHint;
using Arcane::eMemoryResource;
using Arcane::IMemoryAllocator;
using Arcane::IMemoryAllocator3;
using Arcane::MemoryAllocationArgs;
using Arcane::MemoryAllocationOptions;
using Arcane::PrintableMemoryAllocator;
using Arcane::SharedArray;
using Arcane::SharedArray2;
using Arcane::UniqueArray;
using Arcane::UniqueArray2;

//! Typedef pour la version Arcane historique (avec 2's')
// TODO: A rendre obsolète (juin 2025)
using eMemoryRessource = Arcane::eMemoryResource;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

