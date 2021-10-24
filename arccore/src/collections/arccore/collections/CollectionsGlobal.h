// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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
class ArrayMetaData;
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

