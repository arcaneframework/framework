// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BaseTypes.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Definition of types for the 'base' component of Arccore.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASETYPES_H
#define ARCCORE_BASE_BASETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file BaseTypes.h
 *
 * \brief Declarations of types for the 'base' component of %Arccore.
 */

template <typename T> class IterT;
template <typename T> class ConstIterT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type of a byte
typedef unsigned char Byte;
//! Type of a byte
typedef signed char SByte;
//! Type of a unicode character
typedef unsigned short UChar;
//! Type of an unsigned Int16
typedef unsigned short UInt16;
//! Type of a single-precision real number
typedef float Single;

//! Constant to indicate that an array dimension is dynamic
inline constexpr Int32 DynExtent = -1;

template <typename T> class ConstArrayView;
template <typename T> class ArrayView;
template <typename T> class ConstArray2View;
template <typename T> class Array2View;
template <typename T> class ConstArray3View;
template <typename T> class Array3View;
template <typename T> class ConstArray4View;
template <typename T> class Array4View;
template <typename T, typename SizeType, SizeType Extent = DynExtent> class SpanImpl;
template <typename T, Int64 Extent = DynExtent> class Span;
template <typename T, Int32 Extent = DynExtent> class SmallSpan;
template <typename T, Int64 Extent = DynExtent> using LargeSpan = Span<T, Extent>;
template <typename T, typename SizeType, SizeType Extent1 = DynExtent, SizeType Extent2 = DynExtent> class Span2Impl;
template <typename T, Int64 Extent1 = DynExtent, Int64 Extent2 = DynExtent> class Span2;
template <typename T, Int32 Extent1 = DynExtent, Int32 Extent2 = DynExtent> class SmallSpan2;
template <typename Iterator_>
class ArrayIterator;

class StringImpl;
class String;
class StringVector;
class StringBuilder;
class StringFormatterArg;
struct ReferenceCounterTag;

class IRangeFunctor;
template <int RankValue> class IMDRangeFunctor;
template <typename InstanceType> class RangeFunctorT;
template <typename LambdaType> class LambdaRangeFunctorT;
template <typename LambdaType, typename... Views> class LambdaRangeFunctorTVa;
class ForLoopTraceInfo;
template <typename IndexType_ = Int32> class ForLoopRange;
template <int RankValue, typename IndexType_ = Int32> class SimpleForLoopRanges;
template <int RankValue, typename IndexType_ = Int32> class ComplexForLoopRanges;
class ForLoopOneExecStat;
class ForLoopRunInfo;
class ParallelLoopOptions;

class TraceInfo;
class StackTrace;
class Exception;
class StackFrame;
class FixedStackFrameArray;
class IStackTraceService;
class ISymbolizerService;
template <typename T>
class CheckedPointer;
template <class T>
class ReferenceCounterAccessor;
template <typename T>
class ReferenceCounter;
class ReferenceCounterImpl;
class ArgumentException;
class IndexOutOfRangeException;
class FatalErrorException;
class NotSupportedException;
class NotImplementedException;
class TimeoutException;
enum class eBasicDataType : unsigned char;
class IFunctor;
template <typename ArgType>
class IFunctorWithArgumentT;
template <typename ReturnType, typename Arg1, typename Arg2>
class IFunctorWithArgAndReturn2;
template <typename T>
class FunctorT;
template <typename ClassType, typename ArgType>
class FunctorWithArgumentT;
template <typename ArgType>
class StdFunctorWithArgumentT;
class ReferenceCounterImpl;
template <typename InstanceType>
struct RefTraits;
template <typename InstanceType>
class Ref;

namespace impl
{
  template <typename InstanceType>
  class ReferenceCounterWrapper;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// These classes are internal to Arccore/Arcane
template <typename T>
class ArrayRange;
namespace Impl
{
  class BasicTranscoder;
  template <class DataType> class CoreArray;
  class ForLoopStatInfoList;
  class ForLoopStatInfoListImpl;
  class ForLoopCumulativeStat;
} // namespace Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// These classes are not accessible in 'utils' but it is possible
// to use pointers to these instances.
// The definition is in 'arcane_accelerator_core'
namespace Accelerator
{
  class Runner;
  class RunQueue;
} // namespace Accelerator
using Accelerator::Runner;
using Accelerator::RunQueue;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DefaultLayout;
template <int RankValue> class RightLayoutN;
template <int RankValue> class LeftLayoutN;
template <int RankValue> class MDDimType;
class ConstMemoryView;
class MutableMemoryView;
class IMemoryResourceMng;
// TODO: Deprecate
using IMemoryRessourceMng = IMemoryResourceMng;
template <typename IndexType_ = Int32, Int32... RankSize> class ExtentsV;
template <class DataType, typename Extents, typename LayoutPolicy = DefaultLayout>
class MDSpan;
template <typename DataType, typename Extents, typename LayoutPolicy = DefaultLayout>
using MDSpanBase ARCCORE_DEPRECATED_REASON("Use 'MDSpan' type instead") = MDSpan<DataType, Extents, LayoutPolicy>;
template <typename ExtentType> class ArrayBounds;

template <class DataType, typename Extents, typename LayoutType = DefaultLayout>
class NumArray;
template <typename DataType, typename Extents, typename LayoutPolicy = DefaultLayout>
using NumArrayBase ARCCORE_DEPRECATED_REASON("Use 'NumArray' type instead") = NumArray<DataType, Extents, LayoutPolicy>;

template <typename Extents, typename LayoutPolicy> class ArrayExtentsWithOffset;
template <int RankValue, typename IndexType_ = Int32> class MDIndexBase;
template <int RankValue, typename IndexType_ = Int32> class MDIndex;
template <int RankValue, typename IndexType_ = Int32> using ArrayIndexBase = MDIndexBase<RankValue, IndexType_>;
template <int RankValue, typename IndexType_ = Int32> using ArrayIndex = MDIndex<RankValue, IndexType_>;
template <int RankValue> using ArrayBoundsIndexBase ARCCORE_DEPRECATED_REASON("Use 'MDIndexBase' type instead") = ArrayIndexBase<RankValue>;
template <int RankValue> using ArrayBoundsIndex ARCCORE_DEPRECATED_REASON("Use 'MDIndex' type instead") = ArrayIndex<RankValue>;
template <typename Extents> class ArrayExtentsBase;
template <typename Extents> class ArrayExtents;
template <int RankValue> class ArrayStridesBase;
template <int RankValue> class IMDRangeFunctor;
template <int RankValue> class ArrayExtentsValueDynamic;
namespace impl
{
  template <typename IndexType_, Int32... RankSize> class ArrayExtentsValue;
}
template <typename T, Int32 NbElement>
class FixedArray;

class IObservable;
class IObserver;
class Observable;
class Observer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! C equivalent of a one-dimensional array of pointers
typedef ArrayView<Pointer> PointerArrayView;
//! C equivalent of a one-dimensional array of characters
typedef ArrayView<Byte> ByteArrayView;
//! C equivalent of a one-dimensional array of unicode characters
typedef ArrayView<UChar> UCharArrayView;
//! C equivalent of a one-dimensional array of 64-bit integers
typedef ArrayView<Int64> Int64ArrayView;
//! C equivalent of a one-dimensional array of 32-bit integers
typedef ArrayView<Int32> Int32ArrayView;
//! C equivalent of a one-dimensional array of 16-bit integers
typedef ArrayView<Int16> Int16ArrayView;
//! C equivalent of a one-dimensional array of integers
typedef ArrayView<Integer> IntegerArrayView;
//! C equivalent of a one-dimensional array of reals
typedef ArrayView<Real> RealArrayView;
//! C equivalent of a one-dimensional array of booleans
typedef ArrayView<bool> BoolArrayView;
//! C equivalent of a one-dimensional array of integers
typedef ArrayView<Integer> IntegerArrayView;

//! C equivalent of a one-dimensional array of pointers
typedef ConstArrayView<Pointer> PointerConstArrayView;
//! C equivalent of a one-dimensional array of characters
typedef ConstArrayView<Byte> ByteConstArrayView;
//! C equivalent of a one-dimensional array of unicode characters
typedef ConstArrayView<UChar> UCharConstArrayView;
//! C equivalent of a one-dimensional array of 64-bit integers
typedef ConstArrayView<Int64> Int64ConstArrayView;
//! C equivalent of a one-dimensional array of 32-bit integers
typedef ConstArrayView<Int32> Int32ConstArrayView;
//! C equivalent of a one-dimensional array of 16-bit integers
typedef ConstArrayView<Int16> Int16ConstArrayView;
//! C equivalent of a one-dimensional array of integers
typedef ConstArrayView<Integer> IntegerConstArrayView;
//! C equivalent of a one-dimensional array of reals
typedef ConstArrayView<Real> RealConstArrayView;
//! C equivalent of a one-dimensional array of booleans
typedef ConstArrayView<bool> BoolConstArrayView;
//! C equivalent of a one-dimensional array of integers
typedef ConstArrayView<Integer> IntegerConstArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! C equivalent of a one-dimensional array of pointers
typedef Span<Pointer> PointerSpan;
//! C equivalent of a one-dimensional array of characters
typedef Span<std::byte> ByteSpan;
//! C equivalent of a one-dimensional array of unicode characters
typedef Span<UChar> UCharSpan;
//! C equivalent of a one-dimensional array of 64-bit integers
typedef Span<Int64> Int64Span;
//! C equivalent of a one-dimensional array of 32-bit integers
typedef Span<Int32> Int32Span;
//! C equivalent of a one-dimensional array of 16-bit integers
typedef Span<Int16> Int16Span;
//! C equivalent of a one-dimensional array of integers
typedef Span<Integer> IntegerSpan;
//! C equivalent of a one-dimensional array of reals
typedef Span<Real> RealSpan;
//! C equivalent of a one-dimensional array of booleans
typedef Span<bool> BoolSpan;
//! C equivalent of a one-dimensional array of integers
typedef Span<Integer> IntegerSpan;

//! C equivalent of a one-dimensional array of pointers
typedef Span<const Pointer> PointerConstSpan;
//! C equivalent of a one-dimensional array of characters
typedef Span<const std::byte> ByteConstSpan;
//! C equivalent of a one-dimensional array of unicode characters
typedef Span<const UChar> UCharConstSpan;
//! C equivalent of a one-dimensional array of 64-bit integers
typedef Span<const Int64> Int64ConstSpan;
//! C equivalent of a one-dimensional array of 32-bit integers
typedef Span<const Int32> Int32ConstSpan;
//! C equivalent of a one-dimensional array of 16-bit integers
typedef Span<const Int16> Int16ConstSpan;
//! C equivalent of a one-dimensional array of integers
typedef Span<const Integer> IntegerConstSpan;
//! C equivalent of a one-dimensional array of reals
typedef Span<const Real> RealConstSpan;
//! C equivalent of a one-dimensional array of booleans
typedef Span<const bool> BoolConstSpan;
//! C equivalent of a one-dimensional array of integers
typedef Span<const Integer> IntegerConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::Array2View;
using Arcane::Array3View;
using Arcane::Array4View;
using Arcane::ArrayIterator;
using Arcane::ArrayView;
using Arcane::ConstArray2View;
using Arcane::ConstArray3View;
using Arcane::ConstArray4View;
using Arcane::ConstArrayView;
using Arcane::ConstIterT;
using Arcane::eBasicDataType;
using Arcane::IterT;
using Arcane::LargeSpan;
using Arcane::SmallSpan;
using Arcane::SmallSpan2;
using Arcane::Span;
using Arcane::Span2;
using Arcane::Span2Impl;
using Arcane::SpanImpl;
using Arcane::StringBuilder;
using Arcane::StringImpl;
using Arcane::StringVector;

using Arcane::ArgumentException;
using Arcane::CheckedPointer;
using Arcane::Exception;
using Arcane::FatalErrorException;
using Arcane::FixedStackFrameArray;
using Arcane::IndexOutOfRangeException;
using Arcane::IStackTraceService;
using Arcane::NotImplementedException;
using Arcane::NotSupportedException;
using Arcane::Ref;
using Arcane::ReferenceCounter;
using Arcane::ReferenceCounterImpl;
using Arcane::RefTraits;
using Arcane::StackFrame;
using Arcane::StackTrace;
using Arcane::TimeoutException;
using Arcane::TraceInfo;

using Arcane::FunctorT;
using Arcane::FunctorWithArgumentT;
using Arcane::IFunctor;
using Arcane::IFunctorWithArgAndReturn2;
using Arcane::IFunctorWithArgumentT;
using Arcane::StdFunctorWithArgumentT;

using Arcane::Byte;
using Arcane::SByte;
using Arcane::Single;
using Arcane::UChar;
using Arcane::UInt16;

using Arcane::BoolArrayView;
using Arcane::ByteArrayView;
using Arcane::Int16ArrayView;
using Arcane::Int32ArrayView;
using Arcane::Int64ArrayView;
using Arcane::IntegerArrayView;
using Arcane::PointerArrayView;
using Arcane::RealArrayView;
using Arcane::UCharArrayView;

using Arcane::BoolConstArrayView;
using Arcane::ByteConstArrayView;
using Arcane::Int16ConstArrayView;
using Arcane::Int32ConstArrayView;
using Arcane::Int64ConstArrayView;
using Arcane::IntegerConstArrayView;
using Arcane::PointerConstArrayView;
using Arcane::RealConstArrayView;
using Arcane::UCharConstArrayView;

using Arcane::BoolSpan;
using Arcane::ByteSpan;
using Arcane::Int16Span;
using Arcane::Int32Span;
using Arcane::Int64Span;
using Arcane::IntegerSpan;
using Arcane::PointerSpan;
using Arcane::RealSpan;
using Arcane::UCharSpan;

using Arcane::BoolConstSpan;
using Arcane::ByteConstSpan;
using Arcane::Int16ConstSpan;
using Arcane::Int32ConstSpan;
using Arcane::Int64ConstSpan;
using Arcane::IntegerConstSpan;
using Arcane::PointerConstSpan;
using Arcane::RealConstSpan;
using Arcane::UCharConstSpan;

using Arcane::DynExtent;

// These classes are internal to Arccore/Arcane
using Arcane::ArrayRange;
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
