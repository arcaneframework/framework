// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UtilsTypes.h                                                (C) 2000-2022 */
/*                                                                           */
/* Définition des types généraux des classes utilitaires de Arcane.          */
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
 * \brief Déclarations des types utilisés dans Arcane.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classes de 'arccore_base'

using Arccore::ArrayView;
using Arccore::ConstArrayView;
using Arccore::Array2View;
using Arccore::ConstArray2View;
using Arccore::Array3View;
using Arccore::ConstArray3View;
using Arccore::Array4View;
using Arccore::ConstArray4View;
using Arccore::IterT;
using Arccore::ConstIterT;
using Arccore::CoreArray;
using Arccore::StringImpl;
using Arccore::Span;
using Arccore::Span2;
using Arccore::SmallSpan;
using Arccore::SmallSpan2;
using Arccore::SpanImpl;
using Arccore::Span2Impl;
using Arccore::eBasicDataType;

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classes de 'arccore_collections'

using Arccore::IMemoryAllocator;
using Arccore::PrintableMemoryAllocator;
using Arccore::AlignedMemoryAllocator;
using Arccore::DefaultMemoryAllocator;
using Arccore::ArrayImplBase;
using Arccore::ArrayTraits;
using Arccore::ArrayImplT;
using Arccore::Array;
using Arccore::AbstractArray;
using Arccore::SharedArray;
using Arccore::UniqueArray;
using Arccore::Array2;
using Arccore::SharedArray2;
using Arccore::UniqueArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classes de 'arccore_concurrency'

using Arccore::IThreadImplementation;
using Arccore::Mutex;
using Arccore::SpinLock;
using Arccore::GlobalMutex;
using Arccore::IThreadBarrier;
using Arccore::ThreadImpl;
using Arccore::MutexImpl;
using Arccore::NullThreadImplementation;
using Arccore::NullThreadBarrier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classes de 'arccore_trace'

using Arccore::ITraceStream;
using Arccore::ITraceMng;
using Arccore::TraceAccessor;
using Arccore::TraceMessageClass; 
using Arccore::TraceClassConfig;
using Arccore::TraceMessage;
using Arccore::TraceMessageDbg;
using Arccore::TraceMessageListenerArgs;
using Arccore::ITraceMessageListener;
namespace Trace = ::Arccore::Trace;

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

template<typename T> class MultiArray2View;
template<typename T> class ConstMultiArray2View;
template<typename T> class MultiArray2;
template<typename DataType> class UniqueMultiArray2;
template<typename DataType> class SharedMultiArray2;

template<typename T> class EnumeratorT;
template<typename T> class ListEnumeratorT;

template<typename T> class Collection;
template<typename T> class List;

template<typename... Args> class EventObservable;
template<typename... Args> class EventObserver;

class Real2;
class Real3;
class Real2x2;
class Real3x3;
class HPReal;
class JSONWriter;
class JSONValue;
class JSONDocument;
class JSONValueList;
class CommandLineArguments;
class ApplicationInfo;
class VersionInfo;

class IObservable;
class IObserver;
class Observable;
class Observer;

class ArrayShape;
class MemoryView;
class MutableMemoryView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPerformanceCounterService;
class ForLoopTraceInfo;
class ForLoopOneExecStat;
namespace impl
{
class ForLoopStatInfoList;
class ForLoopStatInfoListImpl;
class ForLoopCumulativeStat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Dans ConcurrencyUtils.h

class ParallelLoopOptions;
class ParallelFor1DLoopInfo;
class TaskContext;
class ITaskFunctor;
template<typename InstanceType>
class TaskFunctor;
template<typename InstanceType>
class TaskFunctorWithContext;
class ITask;
class ITaskImplementation;
class TaskFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe pour un tableau dynamique de rang RankValue
template<int RankValue>
class MDDim
{
 public:
  static constexpr int rank() { return RankValue; }
};

// Ces quatres macros pourront être supprimées après la 3.8

// A définir lorsqu'on voudra que le rang des classes NumArray et associées
// soit spécifier par une classe au lieu d'un entier
#define ARCANE_USE_TYPE_FOR_EXTENT
#define A_MDRANK_TYPE(rank_name) typename rank_name
#define A_MDRANK_RANK_VALUE(rank_name) (rank_name :: rank())
#define A_MDDIM(rank_value) MDDim< rank_value >

//! Constante pour un tableau dynamique de rang 0
using MDDim0 = MDDim<0>;
//! Constante pour un tableau dynamique de rang 1
using MDDim1 = MDDim<1>;
//! Constante pour un tableau dynamique de rang 2
using MDDim2 = MDDim<2>;
//! Constante pour un tableau dynamique de rang 3
using MDDim3 = MDDim<3>;
//! Constante pour un tableau dynamique de rang 4
using MDDim4 = MDDim<4>;


enum class eMemoryRessource;
template<A_MDRANK_TYPE(RankValue)> class DefaultLayout;
class IMemoryRessourceMng;
template<typename DataType,A_MDRANK_TYPE(RankValue),typename LayoutType = DefaultLayout<RankValue> >
class MDSpanBase;
template<class DataType,A_MDRANK_TYPE(RankValue),typename LayoutType = DefaultLayout<RankValue> >
class MDSpan;
template<typename DataType,A_MDRANK_TYPE(RankValue),typename LayoutType = DefaultLayout<RankValue> >
class NumArrayBase;
template<class DataType,A_MDRANK_TYPE(RankValue),typename LayoutType = DefaultLayout<RankValue> >
class NumArray;
template<A_MDRANK_TYPE(RankValue)> class ArrayBounds;
template<int RankValue> class ArrayBoundsIndexBase;
template<int RankValue> class ArrayBoundsIndex;
template<A_MDRANK_TYPE(RankValue)> class ArrayExtentsBase;
template<A_MDRANK_TYPE(RankValue)> class ArrayExtents;
template<int RankValue> class ArrayStridesBase;
template<A_MDRANK_TYPE(RankValue),typename LayoutType> class ArrayExtentsWithOffset;
class ForLoopRange;
template<int RankValue> class SimpleForLoopRanges;
template<int RankValue> class ComplexForLoopRanges;
template<int RankValue> class IMDRangeFunctor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour compatibilité avec l'existant
using LoopRange = ForLoopRange;
template<int RankValue> using SimpleLoopRanges = SimpleForLoopRanges<RankValue>;
template<int RankValue> using ComplexLoopRanges = ComplexForLoopRanges<RankValue>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique à une dimension de pointeurs
typedef Array<Pointer> PointerArray;
//! Tableau dynamique à une dimension de caractères
typedef Array<Byte> ByteArray;
//! Tableau dynamique à une dimension de caractères unicode
typedef Array<UChar> UCharArray;
//! Tableau dynamique à une dimension d'entiers 64 bits
typedef Array<Int64> Int64Array;
//! Tableau dynamique à une dimension d'entiers 32 bits
typedef Array<Int32> Int32Array;
//! Tableau dynamique à une dimension d'entiers 16 bits
typedef Array<Int16> Int16Array;
//! Tableau dynamique à une dimension d'entiers
typedef Array<Integer> IntegerArray;
//! Tableau dynamique à une dimension de réels
typedef Array<Real> RealArray;
//! Tableau dynamique à une dimension de booléens
typedef Array<bool> BoolArray;
//! Tableau dynamique à une dimension de chaînes de caractères
typedef Array<String> StringArray;
//! Tableau dynamique à une dimension de vecteurs de rang 2
typedef Array<Real2> Real2Array;
//! Tableau dynamique à une dimension de vecteurs de rang 3
typedef Array<Real3> Real3Array;
//! Tableau dynamique à une dimension de tenseurs de rang 2
typedef Array<Real2x2> Real2x2Array;
//! Tableau dynamique à une dimension de tenseurs de rang 3
typedef Array<Real3x3> Real3x3Array;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique 2D de pointeurs
typedef Array2<Pointer> PointerArray2;
//! Tableau dynamique 2D de caractères
typedef Array2<Byte> ByteArray2;
//! Tableau dynamique 2D de caractères unicode
typedef Array2<UChar> UCharArray2;
//! Tableau dynamique 2D d'entiers 64 bits
typedef Array2<Int64> Int64Array2;
//! Tableau dynamique 2D d'entiers 32 bits
typedef Array2<Int32> Int32Array2;
//! Tableau dynamique 2D d'entiers 16 bits
typedef Array2<Int16> Int16Array2;
//! Tableau dynamique 2D d'entiers
typedef Array2<Integer> IntegerArray2;
//! Tableau dynamique 2D de réels
typedef Array2<Real> RealArray2;
//! Tableau dynamique 2D de booléens
typedef Array2<bool> BoolArray2;
//! Tableau dynamique 2D de chaînes de caractères
typedef Array2<String> StringArray2;
//! Tableau dynamique 2D de vecteurs de rang 2
typedef Array2<Real2> Real2Array2;
//! Tableau dynamique 2D de vecteurs de rang 3
typedef Array2<Real3> Real3Array2;
//! Tableau dynamique 2D de tenseurs de rang 2
typedef Array2<Real2x2> Real2x2Array2;
//! Tableau dynamique 2D de tenseurs de rang 3
typedef Array2<Real3x3> Real3x3Array2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique 2D de pointeurs
typedef SharedArray2<Pointer> PointerSharedArray2;
//! Tableau dynamique 2D de caractères
typedef SharedArray2<Byte> ByteSharedArray2;
//! Tableau dynamique 2D de caractères unicode
typedef SharedArray2<UChar> UCharSharedArray2;
//! Tableau dynamique 2D d'entiers 64 bits
typedef SharedArray2<Int64> Int64SharedArray2;
//! Tableau dynamique 2D d'entiers 32 bits
typedef SharedArray2<Int32> Int32SharedArray2;
//! Tableau dynamique 2D d'entiers
typedef SharedArray2<Integer> IntegerSharedArray2;
//! Tableau dynamique 2D de réels
typedef SharedArray2<Real> RealSharedArray2;
//! Tableau dynamique 2D de booléens
typedef SharedArray2<bool> BoolSharedArray2;
//! Tableau dynamique 2D de chaînes de caractères
typedef SharedArray2<String> StringSharedArray2;
//! Tableau dynamique 2D de vecteurs de rang 2
typedef SharedArray2<Real2> Real2SharedArray2;
//! Tableau dynamique 2D de vecteurs de rang 3
typedef SharedArray2<Real3> Real3SharedArray2;
//! Tableau dynamique 2D de tenseurs de rang 2
typedef SharedArray2<Real2x2> Real2x2SharedArray2;
//! Tableau dynamique 2D de tenseurs de rang 3
typedef SharedArray2<Real3x3> Real3x3SharedArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique 2D de pointeurs
typedef UniqueArray2<Pointer> PointerUniqueArray2;
//! Tableau dynamique 2D de caractères
typedef UniqueArray2<Byte> ByteUniqueArray2;
//! Tableau dynamique 2D de caractères unicode
typedef UniqueArray2<UChar> UCharUniqueArray2;
//! Tableau dynamique 2D d'entiers 64 bits
typedef UniqueArray2<Int64> Int64UniqueArray2;
//! Tableau dynamique 2D d'entiers 32 bits
typedef UniqueArray2<Int32> Int32UniqueArray2;
//! Tableau dynamique 2D d'entiers
typedef UniqueArray2<Integer> IntegerUniqueArray2;
//! Tableau dynamique 2D de réels
typedef UniqueArray2<Real> RealUniqueArray2;
//! Tableau dynamique 2D de booléens
typedef UniqueArray2<bool> BoolUniqueArray2;
//! Tableau dynamique 2D de chaînes de caractères
typedef UniqueArray2<String> StringUniqueArray2;
//! Tableau dynamique 2D de vecteurs de rang 2
typedef UniqueArray2<Real2> Real2UniqueArray2;
//! Tableau dynamique 2D de vecteurs de rang 3
typedef UniqueArray2<Real3> Real3UniqueArray2;
//! Tableau dynamique 2D de tenseurs de rang 2
typedef UniqueArray2<Real2x2> Real2x2UniqueArray2;
//! Tableau dynamique 2D de tenseurs de rang 3
typedef UniqueArray2<Real3x3> Real3x3UniqueArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique 2D de pointeurs
typedef Array2View<Pointer> PointerArray2View;
//! Tableau dynamique 2D de caractères
typedef Array2View<Byte> ByteArray2View;
//! Tableau dynamique 2D de caractères unicode
typedef Array2View<UChar> UCharArray2View;
//! Tableau dynamique 2D d'entiers 64 bits
typedef Array2View<Int64> Int64Array2View;
//! Tableau dynamique 2D d'entiers 32 bits
typedef Array2View<Int32> Int32Array2View;
//! Tableau dynamique 2D d'entiers 16 bits
typedef Array2View<Int16> Int16Array2View;
//! Tableau dynamique 2D d'entiers
typedef Array2View<Integer> IntegerArray2View;
//! Tableau dynamique 2D de réels
typedef Array2View<Real> RealArray2View;
//! Tableau dynamique 2D de booléens
typedef Array2View<bool> BoolArray2View;
//! Tableau dynamique 2D de chaînes de caractères
typedef Array2View<String> StringArray2View;
//! Tableau dynamique 2D de vecteurs de rang 2
typedef Array2View<Real2> Real2Array2View;
//! Tableau dynamique 2D de vecteurs de rang 3
typedef Array2View<Real3> Real3Array2View;
//! Tableau dynamique 2D de tenseurs de rang 2
typedef Array2View<Real2x2> Real2x2Array2View;
//! Tableau dynamique 2D de tenseurs de rang 3
typedef Array2View<Real3x3> Real3x3Array2View;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique 2D de pointeurs
typedef ConstArray2View<Pointer> PointerConstArray2View;
//! Tableau dynamique 2D de caractères
typedef ConstArray2View<Byte> ByteConstArray2View;
//! Tableau dynamique 2D de caractères unicode
typedef ConstArray2View<UChar> UCharConstArray2View;
//! Tableau dynamique 2D d'entiers 64 bits
typedef ConstArray2View<Int64> Int64ConstArray2View;
//! Tableau dynamique 2D d'entiers 32 bits
typedef ConstArray2View<Int32> Int32ConstArray2View;
//! Tableau dynamique 2D d'entiers 16 bits
typedef ConstArray2View<Int16> Int16ConstArray2View;
//! Tableau dynamique 2D d'entiers
typedef ConstArray2View<Integer> IntegerConstArray2View;
//! Tableau dynamique 2D de réels
typedef ConstArray2View<Real> RealConstArray2View;
//! Tableau dynamique 2D de booléens
typedef ConstArray2View<bool> BoolConstArray2View;
//! Tableau dynamique 2D de chaînes de caractères
typedef ConstArray2View<String> StringConstArray2View;
//! Tableau dynamique 2D de vecteurs de rang 2
typedef ConstArray2View<Real2> Real2ConstArray2View;
//! Tableau dynamique 2D de vecteurs de rang 3
typedef ConstArray2View<Real3> Real3ConstArray2View;
//! Tableau dynamique 2D de tenseurs de rang 2
typedef ConstArray2View<Real2x2> Real2x2ConstArray2View;
//! Tableau dynamique 2D de tenseurs de rang 3
typedef ConstArray2View<Real3x3> Real3x3ConstArray2View;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique à une dimension de pointeurs
typedef UniqueArray<Pointer> PointerUniqueArray;
//! Tableau dynamique à une dimension de caractères
typedef UniqueArray<Byte> ByteUniqueArray;
//! Tableau dynamique à une dimension de caractères unicode
typedef UniqueArray<UChar> UCharUniqueArray;
//! Tableau dynamique à une dimension d'entiers 64 bits
typedef UniqueArray<Int64> Int64UniqueArray;
//! Tableau dynamique à une dimension d'entiers 32 bits
typedef UniqueArray<Int32> Int32UniqueArray;
//! Tableau dynamique à une dimension d'entiers 16 bits
typedef UniqueArray<Int16> Int16UniqueArray;
//! Tableau dynamique à une dimension d'entiers
typedef UniqueArray<Integer> IntegerUniqueArray;
//! Tableau dynamique à une dimension de réels
typedef UniqueArray<Real> RealUniqueArray;
//! Tableau dynamique à une dimension de booléens
typedef UniqueArray<bool> BoolUniqueArray;
//! Tableau dynamique à une dimension de chaînes de caractères
typedef UniqueArray<String> StringUniqueArray;
//! Tableau dynamique à une dimension de vecteurs de rang 2
typedef UniqueArray<Real2> Real2UniqueArray;
//! Tableau dynamique à une dimension de vecteurs de rang 3
typedef UniqueArray<Real3> Real3UniqueArray;
//! Tableau dynamique à une dimension de tenseurs de rang 2
typedef UniqueArray<Real2x2> Real2x2UniqueArray;
//! Tableau dynamique à une dimension de tenseurs de rang 3
typedef UniqueArray<Real3x3> Real3x3UniqueArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau dynamique à une dimension de pointeurs
typedef SharedArray<Pointer> PointerSharedArray;
//! Tableau dynamique à une dimension de caractères
typedef SharedArray<Byte> ByteSharedArray;
//! Tableau dynamique à une dimension de caractères unicode
typedef SharedArray<UChar> UCharSharedArray;
//! Tableau dynamique à une dimension d'entiers 64 bits
typedef SharedArray<Int64> Int64SharedArray;
//! Tableau dynamique à une dimension d'entiers 32 bits
typedef SharedArray<Int32> Int32SharedArray;
//! Tableau dynamique à une dimension d'entiers 16 bits
typedef SharedArray<Int16> Int16SharedArray;
//! Tableau dynamique à une dimension d'entiers
typedef SharedArray<Integer> IntegerSharedArray;
//! Tableau dynamique à une dimension de réels
typedef SharedArray<Real> RealSharedArray;
//! Tableau dynamique à une dimension de booléens
typedef SharedArray<bool> BoolSharedArray;
//! Tableau dynamique à une dimension de chaînes de caractères
typedef SharedArray<String> StringSharedArray;
//! Tableau dynamique à une dimension de vecteurs de rang 2
typedef SharedArray<Real2> Real2SharedArray;
//! Tableau dynamique à une dimension de vecteurs de rang 3
typedef SharedArray<Real3> Real3SharedArray;
//! Tableau dynamique à une dimension de tenseurs de rang 2
typedef SharedArray<Real2x2> Real2x2SharedArray;
//! Tableau dynamique à une dimension de tenseurs de rang 3
typedef SharedArray<Real3x3> Real3x3SharedArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableau 2D a taille variable de pointeurs
typedef MultiArray2<Pointer> PointerMultiArray2;
//! Tableau 2D a taille variable de caractères
typedef MultiArray2<Byte> ByteMultiArray2;
//! Tableau 2D a taille variable de caractères unicode
typedef MultiArray2<UChar> UCharMultiArray2;
//! Tableau 2D a taille variable d'entiers 64 bits
typedef MultiArray2<Int64> Int64MultiArray2;
//! Tableau 2D a taille variable d'entiers 32 bits
typedef MultiArray2<Int32> Int32MultiArray2;
//! Tableau 2D a taille variable d'entiers 16 bits
typedef MultiArray2<Int16> Int16MultiArray2;
//! Tableau 2D a taille variable d'entiers
typedef MultiArray2<Integer> IntegerMultiArray2;
//! Tableau 2D a taille variable de réels
typedef MultiArray2<Real> RealMultiArray2;
//! Tableau 2D a taille variable de booléens
typedef MultiArray2<bool> BoolMultiArray2;
//! Tableau 2D a taille variable de chaînes de caractères
typedef MultiArray2<String> StringMultiArray2;
//! Tableau 2D a taille variable de vecteurs de rang 2
typedef MultiArray2<Real2> Real2MultiArray2;
//! Tableau 2D a taille variable de vecteurs de rang 3
typedef MultiArray2<Real3> Real3MultiArray2;
//! Tableau 2D a taille variable de tenseurs de rang 2
typedef MultiArray2<Real2x2> Real2x2MultiArray2;
//! Tableau 2D a taille variable de tenseurs de rang 3
typedef MultiArray2<Real3x3> Real3x3MultiArray2;

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
//! Equivalent C d'un tableau à une dimension de chaînes de caractères
typedef ArrayView<String> StringArrayView;
//! Equivalent C d'un tableau à une dimension de Real2
typedef ArrayView<Real2> Real2ArrayView;
//! Equivalent C d'un tableau à une dimension de Real3
typedef ArrayView<Real3> Real3ArrayView;
//! Equivalent C d'un tableau à une dimension de Real2x2
typedef ArrayView<Real2x2> Real2x2ArrayView;
//! Equivalent C d'un tableau à une dimension de Real3x3
typedef ArrayView<Real3x3> Real3x3ArrayView;

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
//! Equivalent C d'un tableau à une dimension de chaînes de caractères
typedef ConstArrayView<String> StringConstArrayView;
//! Equivalent C d'un tableau à une dimension de Real2
typedef ConstArrayView<Real2> Real2ConstArrayView;
//! Equivalent C d'un tableau à une dimension de Real3
typedef ConstArrayView<Real3> Real3ConstArrayView;
//! Equivalent C d'un tableau à une dimension de Real2x2
typedef ConstArrayView<Real2x2> Real2x2ConstArrayView;
//! Equivalent C d'un tableau à une dimension de Real3x3
typedef ConstArrayView<Real3x3> Real3x3ConstArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Collection de chaînes de caractères. */
typedef Collection<String> StringCollection;

//! Tableau de chaînes de caractères unicode
typedef List<String> StringList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
//! Equivalent C d'un tableau à une dimension de Real2
typedef Span<Real2> Real2Span;
//! Equivalent C d'un tableau à une dimension de Real3
typedef Span<Real3> Real3Span;
//! Equivalent C d'un tableau à une dimension de Real2x2
typedef Span<Real2x2> Real2x2Span;
//! Equivalent C d'un tableau à une dimension de Real3x3
typedef Span<Real3x3> Real3x3Span;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
//! Equivalent C d'un tableau à une dimension de Real2
typedef Span<const Real2> Real2ConstSpan;
//! Equivalent C d'un tableau à une dimension de Real3
typedef Span<const Real3> Real3ConstSpan;
//! Equivalent C d'un tableau à une dimension de Real2x2
typedef Span<const Real2x2> Real2x2ConstSpan;
//! Equivalent C d'un tableau à une dimension de Real3x3
typedef Span<const Real3x3> Real3x3ConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
