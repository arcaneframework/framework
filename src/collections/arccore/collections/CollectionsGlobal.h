// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* CollectionsGlobal.h                                         (C) 2000-2018 */
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

#endif  

