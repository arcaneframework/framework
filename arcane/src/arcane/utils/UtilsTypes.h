// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UtilsTypes.h                                                (C) 2000-2025 */
/*                                                                           */
/* Definition of general types for Arcane utility classes.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_UTILSTYPES_H
#define ARCANE_UTILS_UTILSTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arccore/base/BaseTypes.h"
#include "arccore/base/RefDeclarations.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/concurrency/ConcurrencyGlobal.h"
#include "arccore/trace/TraceGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file UtilsTypes.h
 *
 * \brief Declarations of types used in Arcane.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> class MultiArray2View;
template <typename T> class ConstMultiArray2View;
template <typename T> class MultiArray2;
template <typename DataType> class UniqueMultiArray2;
template <typename DataType> class SharedMultiArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Real2;
class Real3;
class Real2x2;
class Real3x3;
template <typename T> class Vector2;
template <typename T> class Vector3;
using Int64x3 = Vector3<Int64>;
using Int32x3 = Vector3<Int32>;
using Int64x2 = Vector2<Int64>;
using Int32x2 = Vector2<Int32>;
template <typename T, int Size> class NumVector;
template <typename T, int RowSize, int ColumnSize = RowSize> class NumMatrix;
using RealN2 = NumVector<Real, 2>;
using RealN3 = NumVector<Real, 3>;
using RealN2x2 = NumMatrix<Real, 2>;
using RealN3x3 = NumMatrix<Real, 3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HPReal;
class CommandLineArguments;
class ApplicationInfo;
class VersionInfo;

class IObservable;
class IObserver;
class Observable;
class Observer;

class ArrayShape;
using MemoryView ARCANE_DEPRECATED_REASON("Use 'ConstMemoryView' instead") = ConstMemoryView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPerformanceCounterService;
class ForLoopOneExecStat;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// In ConcurrencyUtils.h

class ParallelLoopOptions;
class ParallelFor1DLoopInfo;
class TaskContext;
class ITaskFunctor;
template <typename InstanceType>
class TaskFunctor;
template <typename InstanceType>
class TaskFunctorWithContext;
class ITask;
class ITaskImplementation;
class TaskFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// For compatibility with existing code (to be removed after version 3.8)
template <typename IndexType_ = Int32> using LoopRange = ForLoopRange<IndexType_>;
template <int RankValue> using SimpleLoopRanges = SimpleForLoopRanges<RankValue>;
template <int RankValue> using ComplexLoopRanges = ComplexForLoopRanges<RankValue>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic one-dimensional array of pointers
typedef Array<Pointer> PointerArray;
//! Dynamic one-dimensional array of characters
typedef Array<Byte> ByteArray;
//! Dynamic one-dimensional array of unicode characters
typedef Array<UChar> UCharArray;
//! Dynamic one-dimensional array of 64-bit integers
typedef Array<Int64> Int64Array;
//! Dynamic one-dimensional array of 32-bit integers
typedef Array<Int32> Int32Array;
//! Dynamic one-dimensional array of 16-bit integers
typedef Array<Int16> Int16Array;
//! Dynamic one-dimensional array of 8-bit integers
typedef Array<Int8> Int8Array;
//! Dynamic one-dimensional array of integers
typedef Array<Integer> IntegerArray;
//! Dynamic one-dimensional array of reals
typedef Array<Real> RealArray;
//! Dynamic one-dimensional array of 'bfloat16'
typedef Array<BFloat16> BFloat16Array;
//! Dynamic one-dimensional array of 'float16'
typedef Array<Float16> Float16Array;
//! Dynamic one-dimensional array of 'float'
typedef Array<Float32> Float32Array;
//! Dynamic one-dimensional array of booleans
typedef Array<bool> BoolArray;
//! Dynamic one-dimensional array of strings
typedef Array<String> StringArray;
//! Dynamic one-dimensional array of rank 2 vectors
typedef Array<Real2> Real2Array;
//! Dynamic one-dimensional array of rank 3 vectors
typedef Array<Real3> Real3Array;
//! Dynamic one-dimensional array of rank 2 tensors
typedef Array<Real2x2> Real2x2Array;
//! Dynamic one-dimensional array of rank 3 tensors
typedef Array<Real3x3> Real3x3Array;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 2D array of pointers
typedef Array2<Pointer> PointerArray2;
//! Dynamic 2D array of characters
typedef Array2<Byte> ByteArray2;
//! Dynamic 2D array of unicode characters
typedef Array2<UChar> UCharArray2;
//! Dynamic 2D array of 64-bit integers
typedef Array2<Int64> Int64Array2;
//! Dynamic 2D array of 32-bit integers
typedef Array2<Int32> Int32Array2;
//! Dynamic 2D array of 16-bit integers
typedef Array2<Int16> Int16Array2;
//! Dynamic 2D array of 8-bit integers
typedef Array2<Int8> Int8Array2;
//! Dynamic 2D array of integers
typedef Array2<Integer> IntegerArray2;
//! Dynamic 2D array of reals
typedef Array2<Real> RealArray2;
//! Dynamic 2D array of 'bfloat16'
typedef Array2<BFloat16> BFloat16Array2;
//! Dynamic 2D array of 'float16'
typedef Array2<Float16> Float16Array2;
//! Dynamic 2D array of 'float'
typedef Array2<Float32> Float32Array2;
//! Dynamic 2D array of booleans
typedef Array2<bool> BoolArray2;
//! Dynamic 2D array of strings
typedef Array2<String> StringArray2;
//! Dynamic 2D array of rank 2 vectors
typedef Array2<Real2> Real2Array2;
//! Dynamic 2D array of rank 3 vectors
typedef Array2<Real3> Real3Array2;
//! Dynamic 2D array of rank 2 tensors
typedef Array2<Real2x2> Real2x2Array2;
//! Dynamic 2D array of rank 3 tensors
typedef Array2<Real3x3> Real3x3Array2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 2D array of pointers
typedef SharedArray2<Pointer> PointerSharedArray2;
//! Dynamic 2D array of characters
typedef SharedArray2<Byte> ByteSharedArray2;
//! Dynamic 2D array of unicode characters
typedef SharedArray2<UChar> UCharSharedArray2;
//! Dynamic 2D array of 64-bit integers
typedef SharedArray2<Int64> Int64SharedArray2;
//! Dynamic 2D array of 32-bit integers
typedef SharedArray2<Int32> Int32SharedArray2;
//! Dynamic 2D array of 16-bit integers
typedef SharedArray2<Int16> Int16SharedArray2;
//! Dynamic 2D array of 8-bit integers
typedef SharedArray2<Int8> Int8SharedArray2;
//! Dynamic 2D array of integers
typedef SharedArray2<Integer> IntegerSharedArray2;
//! Dynamic 2D array of reals
typedef SharedArray2<Real> RealSharedArray2;
//! Dynamic 2D array of 'bfloat16'
typedef SharedArray2<BFloat16> BFloat16SharedArray2;
//! Dynamic 2D array of 'float16'
typedef SharedArray2<Float16> Float16SharedArray2;
//! Dynamic 2D array of 'float'
typedef SharedArray2<Float32> Float32SharedArray2;
//! Dynamic 2D array of booleans
typedef SharedArray2<bool> BoolSharedArray2;
//! Dynamic 2D array of strings
typedef SharedArray2<String> StringSharedArray2;
//! Dynamic 2D array of rank 2 vectors
typedef SharedArray2<Real2> Real2SharedArray2;
//! Dynamic 2D array of rank 3 vectors
typedef SharedArray2<Real3> Real3SharedArray2;
//! Dynamic 2D array of rank 2 tensors
typedef SharedArray2<Real2x2> Real2x2SharedArray2;
//! Dynamic 2D array of rank 3 tensors
typedef SharedArray2<Real3x3> Real3x3SharedArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 2D array of pointers
typedef UniqueArray2<Pointer> PointerUniqueArray2;
//! Dynamic 2D array of characters
typedef UniqueArray2<Byte> ByteUniqueArray2;
//! Dynamic 2D array of unicode characters
typedef UniqueArray2<UChar> UCharUniqueArray2;
//! Dynamic 2D array of 64-bit integers
typedef UniqueArray2<Int64> Int64UniqueArray2;
//! Dynamic 2D array of 32-bit integers
typedef UniqueArray2<Int32> Int32UniqueArray2;
//! Dynamic 2D array of integers
typedef UniqueArray2<Integer> IntegerUniqueArray2;
//! Dynamic 2D array of reals
typedef UniqueArray2<Real> RealUniqueArray2;
//! Dynamic 2D array of booleans
typedef UniqueArray2<bool> BoolUniqueArray2;
//! Dynamic 2D array of strings
typedef UniqueArray2<String> StringUniqueArray2;
//! Dynamic 2D array of rank 2 vectors
typedef UniqueArray2<Real2> Real2UniqueArray2;
//! Dynamic 2D array of rank 3 vectors
typedef UniqueArray2<Real3> Real3UniqueArray2;
//! Dynamic 2D array of rank 2 tensors
typedef UniqueArray2<Real2x2> Real2x2UniqueArray2;
//! Dynamic 2D array of rank 3 tensors
typedef UniqueArray2<Real3x3> Real3x3UniqueArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 2D array view of pointers
typedef Array2View<Pointer> PointerArray2View;
//! Dynamic 2D array view of characters
typedef Array2View<Byte> ByteArray2View;
//! Dynamic 2D array view of unicode characters
typedef Array2View<UChar> UCharArray2View;
//! Dynamic 2D array view of 64-bit integers
typedef Array2View<Int64> Int64Array2View;
//! Dynamic 2D array view of 32-bit integers
typedef Array2View<Int32> Int32Array2View;
//! Dynamic 2D array view of 16-bit integers
typedef Array2View<Int16> Int16Array2View;
//! Dynamic 2D array view of integers
typedef Array2View<Integer> IntegerArray2View;
//! Dynamic 2D array view of reals
typedef Array2View<Real> RealArray2View;
//! Dynamic 2D array view of booleans
typedef Array2View<bool> BoolArray2View;
//! Dynamic 2D array view of strings
typedef Array2View<String> StringArray2View;
//! Dynamic 2D array view of rank 2 vectors
typedef Array2View<Real2> Real2Array2View;
//! Dynamic 2D array view of rank 3 vectors
typedef Array2View<Real3> Real3Array2View;
//! Dynamic 2D array view of rank 2 tensors
typedef Array2View<Real2x2> Real2x2Array2View;
//! Dynamic 2D array view of rank 3 tensors
typedef Array2View<Real3x3> Real3x3Array2View;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 2D array view of pointers (const)
typedef ConstArray2View<Pointer> PointerConstArray2View;
//! Dynamic 2D array view of characters (const)
typedef ConstArray2View<Byte> ByteConstArray2View;
//! Dynamic 2D array view of unicode characters (const)
typedef ConstArray2View<UChar> UCharConstArray2View;
//! Dynamic 2D array view of 64-bit integers (const)
typedef ConstArray2View<Int64> Int64ConstArray2View;
//! Dynamic 2D array view of 32-bit integers (const)
typedef ConstArray2View<Int32> Int32ConstArray2View;
//! Dynamic 2D array view of 16-bit integers (const)
typedef ConstArray2View<Int16> Int16ConstArray2View;
//! Dynamic 2D array view of integers (const)
typedef ConstArray2View<Integer> IntegerConstArray2View;
//! Dynamic 2D array view of reals (const)
typedef ConstArray2View<Real> RealConstArray2View;
//! Dynamic 2D array view of booleans (const)
typedef ConstArray2View<bool> BoolConstArray2View;
//! Dynamic 2D array view of strings (const)
typedef ConstArray2View<String> StringConstArray2View;
//! Dynamic 2D array view of rank 2 vectors (const)
typedef ConstArray2View<Real2> Real2ConstArray2View;
//! Dynamic 2D array view of rank 3 vectors (const)
typedef ConstArray2View<Real3> Real3ConstArray2View;
//! Dynamic 2D array of rank 2 tensors
typedef ConstArray2View<Real2x2> Real2x2ConstArray2View;
//! Dynamic 2D array of rank 3 tensors
typedef ConstArray2View<Real3x3> Real3x3ConstArray2View;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 1D array of pointers
typedef UniqueArray<Pointer> PointerUniqueArray;
//! Dynamic 1D array of characters
typedef UniqueArray<Byte> ByteUniqueArray;
//! Dynamic 1D array of unicode characters
typedef UniqueArray<UChar> UCharUniqueArray;
//! Dynamic 1D array of 64-bit integers
typedef UniqueArray<Int64> Int64UniqueArray;
//! Dynamic 1D array of 32-bit integers
typedef UniqueArray<Int32> Int32UniqueArray;
//! Dynamic 1D array of 16-bit integers
typedef UniqueArray<Int16> Int16UniqueArray;
//! Dynamic 1D array of 8-bit integers
typedef UniqueArray<Int8> Int8UniqueArray;
//! Dynamic 1D array of integers
typedef UniqueArray<Integer> IntegerUniqueArray;
//! Dynamic 1D array of reals
typedef UniqueArray<Real> RealUniqueArray;
//! Dynamic 1D array of 'bfloat16'
typedef UniqueArray<BFloat16> BFloat16UniqueArray;
//! Dynamic 1D array of 'float16'
typedef UniqueArray<Float16> Float16UniqueArray;
//! Dynamic 1D array of 'float'
typedef UniqueArray<Float32> Float32UniqueArray;
//! Dynamic 1D array of booleans
typedef UniqueArray<bool> BoolUniqueArray;
//! Dynamic 1D array of strings
typedef UniqueArray<String> StringUniqueArray;
//! Dynamic 1D array of rank 2 vectors
typedef UniqueArray<Real2> Real2UniqueArray;
//! Dynamic 1D array of rank 3 vectors
typedef UniqueArray<Real3> Real3UniqueArray;
//! Dynamic 1D array of rank 2 tensors
typedef UniqueArray<Real2x2> Real2x2UniqueArray;
//! Dynamic 1D array of rank 3 tensors
typedef UniqueArray<Real3x3> Real3x3UniqueArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Dynamic 1D array of pointers
typedef SharedArray<Pointer> PointerSharedArray;
//! Dynamic 1D array of characters
typedef SharedArray<Byte> ByteSharedArray;
//! Dynamic 1D array of unicode characters
typedef SharedArray<UChar> UCharSharedArray;
//! Dynamic 1D array of 64-bit integers
typedef SharedArray<Int64> Int64SharedArray;
//! Dynamic 1D array of 32-bit integers
typedef SharedArray<Int32> Int32SharedArray;
//! Dynamic 1D array of 16-bit integers
typedef SharedArray<Int16> Int16SharedArray;
//! Dynamic 1D array of 8-bit integers
typedef SharedArray<Int8> Int8SharedArray;
//! Dynamic 1D array of integers
typedef SharedArray<Integer> IntegerSharedArray;
//! Dynamic 1D array of reals
typedef SharedArray<Real> RealSharedArray;
//! Dynamic 1D array of 'bfloat16'
typedef SharedArray<BFloat16> BFloat16SharedArray;
//! Dynamic 1D array of 'float16'
typedef SharedArray<Float16> Float16SharedArray;
//! Dynamic 1D array of 'float'
typedef SharedArray<Float32> Float32SharedArray;
//! Dynamic 1D array of booleans
typedef SharedArray<bool> BoolSharedArray;
//! Dynamic 1D array of strings
typedef SharedArray<String> StringSharedArray;
//! Dynamic 1D array of rank 2 vectors
typedef SharedArray<Real2> Real2SharedArray;
//! Dynamic 1D array of rank 3 vectors
typedef SharedArray<Real3> Real3SharedArray;
//! Dynamic 1D array of rank 2 tensors
typedef SharedArray<Real2x2> Real2x2SharedArray;
//! Dynamic 1D array of rank 3 tensors
typedef SharedArray<Real3x3> Real3x3SharedArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! 2D variable size array of pointers
typedef MultiArray2<Pointer> PointerMultiArray2;
//! 2D variable size array of characters
typedef MultiArray2<Byte> ByteMultiArray2;
//! 2D variable size array of unicode characters
typedef MultiArray2<UChar> UCharMultiArray2;
//! 2D variable size array of 64-bit integers
typedef MultiArray2<Int64> Int64MultiArray2;
//! 2D variable size array of 32-bit integers
typedef MultiArray2<Int32> Int32MultiArray2;
//! 2D variable size array of 16-bit integers
typedef MultiArray2<Int16> Int16MultiArray2;
//! 2D variable size array of integers
typedef MultiArray2<Integer> IntegerMultiArray2;
//! 2D variable size array of reals
typedef MultiArray2<Real> RealMultiArray2;
//! 2D variable size array of booleans
typedef MultiArray2<bool> BoolMultiArray2;
//! 2D variable size array of strings
typedef MultiArray2<String> StringMultiArray2;
//! 2D variable size array of rank 2 vectors
typedef MultiArray2<Real2> Real2MultiArray2;
//! 2D variable size array of rank 3 vectors
typedef MultiArray2<Real3> Real3MultiArray2;
//! 2D variable size array of rank 2 tensors
typedef MultiArray2<Real2x2> Real2x2MultiArray2;
//! 2D variable size array of rank 3 tensors
typedef MultiArray2<Real3x3> Real3x3MultiArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! C equivalent of a 1D array of pointers
typedef ArrayView<Pointer> PointerArrayView;
//! C equivalent of a 1D array of characters
typedef ArrayView<Byte> ByteArrayView;
//! C equivalent of a 1D array of unicode characters
typedef ArrayView<UChar> UCharArrayView;
//! C equivalent of a 1D array of 64-bit integers
typedef ArrayView<Int64> Int64ArrayView;
//! C equivalent of a 1D array of 32-bit integers
typedef ArrayView<Int32> Int32ArrayView;
//! C equivalent of a 1D array of 16-bit integers
typedef ArrayView<Int16> Int16ArrayView;
//! C equivalent of a 1D array of integers
typedef ArrayView<Integer> IntegerArrayView;
//! C equivalent of a 1D array of reals
typedef ArrayView<Real> RealArrayView;
//! C equivalent of a 1D array of booleans
typedef ArrayView<bool> BoolArrayView;
//! C equivalent of a 1D array of strings
typedef ArrayView<String> StringArrayView;
//! C equivalent of a 1D array of Real2
typedef ArrayView<Real2> Real2ArrayView;
//! C equivalent of a 1D array of Real3
typedef ArrayView<Real3> Real3ArrayView;
//! C equivalent of a 1D array of Real2x2
typedef ArrayView<Real2x2> Real2x2ArrayView;
//! C equivalent of a 1D array of Real3x3
typedef ArrayView<Real3x3> Real3x3ArrayView;

//! C equivalent of a 1D array of pointers
typedef ConstArrayView<Pointer> PointerConstArrayView;
//! C equivalent of a 1D array of characters
typedef ConstArrayView<Byte> ByteConstArrayView;
//! C equivalent of a 1D array of unicode characters
typedef ConstArrayView<UChar> UCharConstArrayView;
//! C equivalent of a 1D array of 64-bit integers
typedef ConstArrayView<Int64> Int64ConstArrayView;
//! C equivalent of a 1D array of 32-bit integers
typedef ConstArrayView<Int32> Int32ConstArrayView;
//! C equivalent of a 1D array of 16-bit integers
typedef ConstArrayView<Int16> Int16ConstArrayView;
//! C equivalent of a 1D array of integers
typedef ConstArrayView<Integer> IntegerConstArrayView;
//! C equivalent of a 1D array of reals
typedef ConstArrayView<Real> RealConstArrayView;
//! C equivalent of a 1D array of booleans
typedef ConstArrayView<bool> BoolConstArrayView;
//! C equivalent of a 1D array of strings
typedef ConstArrayView<String> StringConstArrayView;
//! C equivalent of a 1D array of Real2
typedef ConstArrayView<Real2> Real2ConstArrayView;
//! C equivalent of a 1D array of Real3
typedef ConstArrayView<Real3> Real3ConstArrayView;
//! C equivalent of a 1D array of Real2x2
typedef ConstArrayView<Real2x2> Real2x2ConstArrayView;
//! C equivalent of a 1D array of Real3x3
typedef ConstArrayView<Real3x3> Real3x3ConstArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Collection of strings. */
typedef Collection<String> StringCollection;

//! Unicode string list
typedef List<String> StringList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef Span<Pointer> PointerSpan;
//! C equivalent of a 1D array of characters
typedef Span<std::byte> ByteSpan;
//! C equivalent of a 1D array of unicode characters
typedef Span<UChar> UCharSpan;
//! C equivalent of a 1D array of 64-bit integers
typedef Span<Int64> Int64Span;
//! C equivalent of a 1D array of 32-bit integers
typedef Span<Int32> Int32Span;
//! C equivalent of a 1D array of 16-bit integers
typedef Span<Int16> Int16Span;
//! C equivalent of a 1D array of integers
typedef Span<Integer> IntegerSpan;
//! C equivalent of a 1D array of reals
typedef Span<Real> RealSpan;
//! C equivalent of a 1D array of booleans
typedef Span<bool> BoolSpan;
//! C equivalent of a 1D array of integers
typedef Span<Integer> IntegerSpan;
//! C equivalent of a 1D array of Real2
typedef Span<Real2> Real2Span;
//! C equivalent of a 1D array of Real3
typedef Span<Real3> Real3Span;
//! C equivalent of a 1D array of Real2x2
typedef Span<Real2x2> Real2x2Span;
//! C equivalent of a 1D array of Real3x3
typedef Span<Real3x3> Real3x3Span;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Read-only view of a 1D array of pointers
typedef Span<const Pointer> PointerConstSpan;
//! Read-only view of a 1D array of characters
typedef Span<const std::byte> ByteConstSpan;
//! Read-only view of a 1D array of unicode characters
typedef Span<const UChar> UCharConstSpan;
//! Read-only view of a 1D array of 64-bit integers
typedef Span<const Int64> Int64ConstSpan;
//! Read-only view of a 1D array of 32-bit integers
typedef Span<const Int32> Int32ConstSpan;
//! Read-only view of a 1D array of 16-bit integers
typedef Span<const Int16> Int16ConstSpan;
//! Read-only view of a 1D array of integers
typedef Span<const Integer> IntegerConstSpan;
//! Read-only view of a 1D array of reals
typedef Span<const Real> RealConstSpan;
//! Read-only view of a 1D array of booleans
typedef Span<const bool> BoolConstSpan;
//! Read-only view of a 1D array of integers
typedef Span<const Integer> IntegerConstSpan;
//! Read-only view of a 1D array of Real2
typedef Span<const Real2> Real2ConstSpan;
//! Read-only view of a 1D array of Real3
typedef Span<const Real3> Real3ConstSpan;
//! Read-only view of a 1D array of Real2x2
typedef Span<const Real2x2> Real2x2ConstSpan;
//! Read-only view of a 1D array of Real3x3
typedef Span<const Real3x3> Real3x3ConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! View of a 1D array of pointers
using PointerSmallSpan = SmallSpan<Pointer>;
//! View of a 1D array of characters
using ByteSmallSpan = SmallSpan<std::byte>;
//! View of a 1D array of unicode characters
using UCharSmallSpan = SmallSpan<UChar>;
//! View of a 1D array of 64-bit integers
using Int64SmallSpan = SmallSpan<Int64>;
//! View of a 1D array of 32-bit integers
using Int32SmallSpan = SmallSpan<Int32>;
//! View of a 1D array of 16-bit integers
using Int16SmallSpan = SmallSpan<Int16>;
//! View of a 1D array of integers
using IntegerSmallSpan = SmallSpan<Integer>;
//! View of a 1D array of reals
using RealSmallSpan = SmallSpan<Real>;
//! View of a 1D array of booleans
using BoolSmallSpan = SmallSpan<bool>;
//! View of a 1D array of integers
using IntegerSmallSpan = SmallSpan<Integer>;
//! View of a 1D array of Real2
using Real2SmallSpan = SmallSpan<Real2>;
//! View of a 1D array of Real3
using Real3SmallSpan = SmallSpan<Real3>;
//! View of a 1D array of Real2x2
using Real2x2SmallSpan = SmallSpan<Real2x2>;
//! View of a 1D array of Real3x3
using Real3x3SmallSpan = SmallSpan<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Read-only view of a 1D array of pointers
using PointerConstSmallSpan = SmallSpan<const Pointer>;
//! Read-only view of a 1D array of characters
using ByteConstSmallSpan = SmallSpan<const std::byte>;
//! Read-only view of a 1D array of unicode characters
using UCharConstSmallSpan = SmallSpan<const UChar>;
//! Read-only view of a 1D array of 64-bit integers
using Int64ConstSmallSpan = SmallSpan<const Int64>;
//! Read-only view of a 1D array of 32-bit integers
using Int32ConstSmallSpan = SmallSpan<const Int32>;
//! Read-only view of a 1D array of 16-bit integers
using Int16ConstSmallSpan = SmallSpan<const Int16>;
//! Read-only view of a 1D array of integers
using IntegerConstSmallSpan = SmallSpan<const Integer>;
//! Read-only view of a 1D array of reals
using RealConstSmallSpan = SmallSpan<const Real>;
//! Read-only view of a 1D array of booleans
using BoolConstSmallSpan = SmallSpan<const bool>;
//! Read-only view of a 1D array of integers
using IntegerConstSmallSpan = SmallSpan<const Integer>;
//! Read-only view of a 1D array of Real2
using Real2ConstSmallSpan = SmallSpan<const Real2>;
//! Read-only view of a 1D array of Real3
using Real3ConstSmallSpan = SmallSpan<const Real3>;
//! Read-only view of a 1D array of Real2x2
using Real2x2ConstSmallSpan = SmallSpan<const Real2x2>;
//! Read-only view of a 1D array of Real3x3
using Real3x3ConstSmallSpan = SmallSpan<const Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
