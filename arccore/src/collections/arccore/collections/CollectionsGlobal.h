// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CollectionsGlobal.h                                         (C) 2000-2023 */
/*                                                                           */
/* Définitions globales de la composante 'Collections' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_COLLECTIONSGLOBAL_H
#define ARCCORE_COLLECTIONS_COLLECTIONSGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

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

namespace Arccore
{
class IMemoryAllocator;
class PrintableMemoryAllocator;
class AlignedMemoryAllocator;
class DefaultMemoryAllocator;
class ArrayImplBase;
class ArrayMetaData;
class MemoryAllocationArgs;
class MemoryAllocationOptions;
template<typename DataType> class ArrayTraits;
template<typename DataType> class ArrayImplT;
template<typename DataType> class Array;
template<typename DataType> class AbstractArray;
template<typename DataType> class UniqueArray;
template<typename DataType> class SharedArray;
template<typename DataType> class Array2;
template<typename DataType> class UniqueArray2;
template<typename DataType> class SharedArray2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

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

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

