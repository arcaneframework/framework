// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BaseTypes.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Définition des types de la composante 'base' de Arccore.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASETYPES_H
#define ARCCORE_BASE_BASETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file BaseTypes.h
 *
 * \brief Déclarations des types de la composante 'base' de %Arccore.
 */

template <typename T> class IterT;
template <typename T> class ConstIterT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type d'un octet
typedef unsigned char Byte;
//! Type d'un octet
typedef signed char SByte;
//! Type d'un caractère unicode
typedef unsigned short UChar;
//! Type d'un Int16 non signé
typedef unsigned short UInt16;
//! Type d'un réel simple précision
typedef float Single;

//! Indique que la dimension d'un tableau est dynamique
inline constexpr Int32 DynExtent = -1;

template <typename T> class ConstArrayView;
template <typename T> class ArrayView;
template <typename T> class ConstArray2View;
template <typename T> class Array2View;
template <typename T> class ConstArray3View;
template <typename T> class Array3View;
template <typename T> class ConstArray4View;
template <typename T> class Array4View;
template <class DataType> class CoreArray;
template <typename T, typename SizeType, SizeType Extent = DynExtent, SizeType MinValue = 0> class SpanImpl;
template <typename T, Int64 Extent = DynExtent, Int64 MinValue = 0> class Span;
template <typename T, Int32 Extent = DynExtent, Int32 MinValue = 0> class SmallSpan;
template <typename T, Int64 Extent = DynExtent, Int64 MinValue = 0> using LargeSpan = Span<T, Extent, MinValue>;
template <typename T, typename SizeType, SizeType Extent1 = DynExtent, SizeType Extent2 = DynExtent> class Span2Impl;
template <typename T, Int64 Extent1 = DynExtent, Int64 Extent2 = DynExtent> class Span2;
template <typename T, Int32 Extent1 = DynExtent, Int32 Extent2 = DynExtent> class SmallSpan2;
template <typename Iterator_>
class ArrayIterator;

class StringImpl;
class String;
class StringBuilder;
class StringFormatterArg;
struct ReferenceCounterTag;

class TraceInfo;
class StackTrace;
class Exception;
class StackFrame;
class FixedStackFrameArray;
class IStackTraceService;
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
template <typename InstanceType, class T = void>
struct RefTraits;
template <typename InstanceType, int ImplTagId = RefTraits<InstanceType>::TagId>
class Ref;

namespace impl
{
  template <typename InstanceType>
  class ReferenceCounterWrapper;
}

// Ces classes sont internes à Arccore/Arcane
template <typename T>
class ArrayRange;
class BasicTranscoder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Equivalent C d'un tableau à une dimension de pointeurs
typedef ArrayView<Pointer> PointerArrayView;
//! Equivalent C d'un tableau à une dimension de caractères
typedef ArrayView<Byte> ByteArrayView;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef ArrayView<UChar> UCharArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef ArrayView<Int64> Int64ArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef ArrayView<Int32> Int32ArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef ArrayView<Int16> Int16ArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ArrayView<Integer> IntegerArrayView;
//! Equivalent C d'un tableau à une dimension de réels
typedef ArrayView<Real> RealArrayView;
//! Equivalent C d'un tableau à une dimension de booléens
typedef ArrayView<bool> BoolArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ArrayView<Integer> IntegerArrayView;

//! Equivalent C d'un tableau à une dimension de pointeurs
typedef ConstArrayView<Pointer> PointerConstArrayView;
//! Equivalent C d'un tableau à une dimension de caractères
typedef ConstArrayView<Byte> ByteConstArrayView;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef ConstArrayView<UChar> UCharConstArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef ConstArrayView<Int64> Int64ConstArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef ConstArrayView<Int32> Int32ConstArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef ConstArrayView<Int16> Int16ConstArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ConstArrayView<Integer> IntegerConstArrayView;
//! Equivalent C d'un tableau à une dimension de réels
typedef ConstArrayView<Real> RealConstArrayView;
//! Equivalent C d'un tableau à une dimension de booléens
typedef ConstArrayView<bool> BoolConstArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ConstArrayView<Integer> IntegerConstArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Equivalent C d'un tableau à une dimension de pointeurs
typedef Span<Pointer> PointerSpan;
//! Equivalent C d'un tableau à une dimension de caractères
typedef Span<std::byte> ByteSpan;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef Span<UChar> UCharSpan;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef Span<Int64> Int64Span;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef Span<Int32> Int32Span;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef Span<Int16> Int16Span;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef Span<Integer> IntegerSpan;
//! Equivalent C d'un tableau à une dimension de réels
typedef Span<Real> RealSpan;
//! Equivalent C d'un tableau à une dimension de booléens
typedef Span<bool> BoolSpan;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef Span<Integer> IntegerSpan;

//! Equivalent C d'un tableau à une dimension de pointeurs
typedef Span<const Pointer> PointerConstSpan;
//! Equivalent C d'un tableau à une dimension de caractères
typedef Span<const std::byte> ByteConstSpan;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef Span<const UChar> UCharConstSpan;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef Span<const Int64> Int64ConstSpan;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef Span<const Int32> Int32ConstSpan;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef Span<const Int16> Int16ConstSpan;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef Span<const Integer> IntegerConstSpan;
//! Equivalent C d'un tableau à une dimension de réels
typedef Span<const Real> RealConstSpan;
//! Equivalent C d'un tableau à une dimension de booléens
typedef Span<const bool> BoolConstSpan;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef Span<const Integer> IntegerConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

namespace Arcane
{
using Arccore::Array2View;
using Arccore::Array3View;
using Arccore::Array4View;
using Arccore::ArrayIterator;
using Arccore::ArrayView;
using Arccore::ConstArray2View;
using Arccore::ConstArray3View;
using Arccore::ConstArray4View;
using Arccore::ConstArrayView;
using Arccore::ConstIterT;
using Arccore::CoreArray;
using Arccore::eBasicDataType;
using Arccore::IterT;
using Arccore::SmallSpan;
using Arccore::SmallSpan2;
using Arccore::Span;
using Arccore::Span2;
using Arccore::Span2Impl;
using Arccore::SpanImpl;
using Arccore::StringImpl;

using Arccore::TraceInfo;
using Arccore::StackTrace;
using Arccore::Exception;
using Arccore::StackFrame;
using Arccore::FixedStackFrameArray;
using Arccore::IStackTraceService;
using Arccore::CheckedPointer;
using Arccore::ReferenceCounter;
using Arccore::ReferenceCounterImpl;
using Arccore::RefTraits;
using Arccore::Ref;
using Arccore::ArgumentException;
using Arccore::IndexOutOfRangeException;
using Arccore::FatalErrorException;
using Arccore::NotSupportedException;
using Arccore::NotImplementedException;
using Arccore::TimeoutException;

using Arccore::IFunctor;
using Arccore::IFunctorWithArgumentT;
using Arccore::IFunctorWithArgAndReturn2;
using Arccore::FunctorT;
using Arccore::FunctorWithArgumentT;
using Arccore::StdFunctorWithArgumentT;

using Arccore::Byte;
using Arccore::SByte;
using Arccore::Single;
using Arccore::UChar;
using Arccore::UInt16;

// Ces classes sont internes à Arccore/Arcane
using Arccore::BasicTranscoder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
