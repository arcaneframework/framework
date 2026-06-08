// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonGlobal.h                                              (C) 2000-2026 */
/*                                                                           */
/* Global definitions for the 'Common' component of 'Arccore'.               */
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

// For now, ArrayTraits must remain in the Arccore namespace
// for compatibility reasons with the ARCCORE_DEFINE_ARRAY_PODTYPE macro
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
class IMemoryPool;

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

template <typename T> class EnumeratorT;
template <typename T> class ListEnumeratorT;

template <typename T> class Collection;
template <typename T> class List;

class EventObservableBase;
class EventObserverBase;
template <typename... Args> class EventObservable;
template <typename... Args> class EventObserver;
template <typename... Args> class EventObservableView;

using StringList = List<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class JSONWriter;
class JSONValue;
class JSONDocument;
class JSONValueList;
class JSONWrapperUtils;
class JSONKeyValue;
class JSONKeyValueList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class DataViewSetter;
template <typename DataType>
class DataViewGetter;
template <typename DataType>
class DataViewGetterSetter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CommandLineArguments;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indices for expected memory location
enum class eMemoryLocationHint : int8_t
{
  //! No hint
  None = 0,
  //! Indicates that the data will primarily be used on the accelerator
  MainlyDevice = 1,
  //! Indicates that the data will primarily be used on the CPU
  MainlyHost = 2,
  /*!
   * \brief Indicates that the data will be used both on the accelerator and
   * on the CPU and will not be frequently modified.
   */
  HostAndDeviceMostlyRead = 3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Physical location of a memory address.
 *
 * For the ManagedMemoryDevice and ManagedMemoryHost values, this is an
 * indication because there is no simple way to know where
 * the memory is actually located.
 */
enum class eHostDeviceMemoryLocation : int8_t
{
  //! Unknown location
  Unknown = 0,
  //! The memory is on the accelerator
  Device = 1,
  //! The memory is on the host.
  Host = 2,
  //! The memory is managed memory on the accelerator
  ManagedMemoryDevice = 3,
  //! The memory is managed memory on the host.
  ManagedMemoryHost = 4,
};

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eHostDeviceMemoryLocation r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of available memory resources.
 */
enum class eMemoryResource
{
  //! Unknown or uninitialized value
  Unknown = 0,
  //! Allocates on the host.
  Host,
  //! Allocates on the host.
  HostPinned,
  //! Allocates on the device
  Device,
  //! Allocates using unified memory.
  UnifiedMemory
};

//! Number of valid values for eMemoryResource
static constexpr int ARCCORE_NB_MEMORY_RESOURCE = 5;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryResource r);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Padding size for indices in SIMD operations.
 *
 * To ensure the same code regardless of the vectorization mechanism
 * used, this value is fixed and corresponds to the largest SIMD vector.
 *
 * \sa arcanedoc_simd
 */
static const Integer SIMD_PADDING_SIZE = 8;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Impl
{
  class StridedLoopRangesBase;
  template <typename LoopRangesType>
  class StridedLoopRanges;
}; // namespace Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
