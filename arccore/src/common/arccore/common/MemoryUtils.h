// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Memory management utility functions.                                      */
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
 * \brief Specific allocator for accelerators.
 *
 * \deprecated Use MemoryUtils::getDefaultDataAllocator() instead.
 */
extern "C++" ARCCORE_DEPRECATED_REASON("Y2024: Use getDefaultDataAllocator() instead.")
ARCCORE_COMMON_EXPORT IMemoryAllocator* getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory resource used by the default allocator for data.
 *
 * By default, if an accelerator runtime is initialized, the associated resource
 * is eMemoryResource::UnifiedMemory. Otherwise, it is eMemoryResource::Host.
 *
 * \sa getDefaultDataAllocator();
 */
extern "C++" ARCCORE_COMMON_EXPORT eMemoryResource
getDefaultDataMemoryResource();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the memory resource by its name.
 *
 * The name corresponds to the name of the enumeration value (e.g.,
 * 'Device' for eMemoryResource::Device.
 *
 * If \a name is null, returns eMemoryResource::Unknown.
 * If \a name does not correspond to a valid value, throws an exception.
 */
extern "C++" ARCCORE_COMMON_EXPORT eMemoryResource
getMemoryResourceFromName(const String& name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocator for data.
 *
 * The default allocator for data is an allocator that allows access to the
 * memory region both by the host and the accelerator.
 *
 * It is possible to retrieve the associated memory resource via
 * getDefaultDataMemoryResource();
 *
 * This call is equivalent to getAllocator(getDefaultDataMemoryResource()).
 *
 * It is guaranteed that the alignment is at least that returned by
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
getDefaultDataAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocator for data with expected location information.
 *
 * This function returns the allocator of getDefaulDataAllocator() but
 * adds the memory management information specified by \a hint.
 */
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
getDefaultDataAllocator(eMemoryLocationHint hint);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the host or device allocator.
 *
 * If an accelerator runtime is initialized, the returned allocator allows
 * allocation using the default accelerator memory (eMemoryResource::Device).
 * Otherwise, it uses the host allocator (eMemoryResource::Host).
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
getDeviceOrHostAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocator for mostly read-only data.
 *
 * This call is equivalent to
 * getDefaultDataAllocator(eMemoryLocationHint::HostAndDeviceMostlyRead).
 */
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
getAllocatorForMostlyReadOnlyData();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocation for the resource \a mem_resource.
 *
 * Throws an exception if no allocator is available for the resource
 * (for example, if eMemoryResource::Device is requested and there is no
 * support for accelerators.
 *
 * The eMemoryResource::UnifiedMemory resource is always available. If
 * no accelerator runtime is loaded, then it is equivalent to
 * eMemoryResource::Host.
 */
extern "C++" ARCCORE_COMMON_EXPORT MemoryAllocationOptions
getAllocationOptions(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Default allocator for the resource \a mem_resource.
 *
 * Throws an exception if no allocator is available for the
 * resource \a mem_resource.
 *
 * \sa getAllocationOptions().
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
getAllocator(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory pool for the resource \a mem_resource.
 *
 * Returns \a nullptr if no memory pool is available for
 * the resource \a mem_resource.
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryPool*
getMemoryPoolOrNull(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies \a source to \a destination using the queue \a queue.
 *
 * It is possible to specify the memory resource where the source
 * and destination are located. If they are unknown, it is preferable to use
 * the overload
 * copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue).
 */
extern "C++" ARCCORE_COMMON_EXPORT void
copy(MutableMemoryView destination, eMemoryResource destination_mem,
     ConstMemoryView source, eMemoryResource source_mem,
     const RunQueue* queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copies \a source to \a destination using the queue \a queue.
inline void
copy(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue = nullptr)
{
  eMemoryResource mem_type = eMemoryResource::Unknown;
  copy(destination, mem_type, source, mem_type, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copies \a source to \a destination using the queue \a queue.
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

//! Copies \a source to \a destination using the queue \a queue.
template <typename DataType> inline void
copy(SmallSpan<DataType> destination, SmallSpan<const DataType> source,
     const RunQueue* queue = nullptr)
{
  copy(Span<DataType>(destination), Span<const DataType>(source), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies data on the host with indirection.
 *
 * Copies the data from \a source into \a destination indexed by \a indexes
 *
 * The operation is equivalent to the following pseudo-code:
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
 * \brief Copies data on the host with indirection.
 *
 * Copies the data from \a source into \a destination indexed by \a indexes
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destination[i] = source[indexes[i]];
 * \endcode
 *
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Copies the data from \a source into \a destination.
 *
 * Uses std::memmove for the copy.
 *
 * \pre source.bytes.size() >= destination.bytes.size()
 */
extern "C++" ARCCORE_COMMON_EXPORT void
copyHost(MutableMemoryView destination, ConstMemoryView source);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies indexed data from \a v into the instance.
 *
 * The operation is equivalent to the following pseudo-code:
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
 * \brief Memory copy with indirection
 *
 * Copies the data from \a source into \a destination for the indices
 * specified by \a indexes.
 *
 * The operation is equivalent to the following pseudo-code:
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destination[indexes[i]] = source[i];
 * \endcode
 *
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Fills an indexed memory region with a value.
 *
 * Fills the indices \a indexes of the memory region \a destination with
 * the value of the memory region \a source. \a source must have a single value.
 * The memory region \a source must be accessible from the host.
 *
 * The operation is equivalent to the following pseudo-code:
 *
 * \code
 * Int32 n = indexes.size();
 * for( Int32 i=0; i<n; ++i )
 *   destination[indexes[i]] = source[0];
 * \endcode
 *
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Fills a memory region with a value.
 *
 * Fills the values of the memory region \a destination with
 * the value of the memory region \a source. \a source must have a single value.
 * The memory region \a source must be accessible from the host.
 *
 * The operation is equivalent to the following pseudo-code:
 *
 * \code
 * Int32 n = nbElement();
 * for( Int32 i=0; i<n; ++i )
 *   destination[i] = source[0];
 * \endcode
 *
 * If \a run_queue is not null, it will be used for the copy.
 *
 * \pre destination.datatypeSize() == source.datatypeSize();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
fill(MutableMemoryView destination, ConstMemoryView source,
     const RunQueue* run_queue = nullptr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies indexed data from \a source into \a destination.
 *
 * The operation is equivalent to the following pseudo-code:
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
 * The array \a indexes must have a size that is a multiple of 2.
 * Even values are used to index the first array and odd values the second.
 *
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Copies indexed elements of \a destination with data from \a source.
 *
 * The operation is equivalent to the following pseudo-code:
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
 * The array \a indexes must have a size that is a multiple of 2.
 * Even values are used to index the first array and odd values the second.
 *
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Fills indexed elements of \a destination with data \a source.
 *
 * \a source must have a single value. This value will be used
 * to fill the values of the instance at the indices specified by
 * \a indexes. It must be accessible from the host.
 *
 * The operation is equivalent to the following pseudo-code:
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
 * If \a run_queue is not null, it will be used for the copy.
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
 * \brief Fills elements of \a destination with the value \a source.
 *
 * \a source must have a single value. It must be accessible from the host.
 *
 * The operation is equivalent to the following pseudo-code:
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
 * If \a run_queue is not null, it will be used for the copy.
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
