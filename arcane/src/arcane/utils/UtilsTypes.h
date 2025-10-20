// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UtilsTypes.h                                                (C) 2000-2025 */
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

template<typename T> class MultiArray2View;
template<typename T> class ConstMultiArray2View;
template<typename T> class MultiArray2;
template<typename DataType> class UniqueMultiArray2;
template<typename DataType> class SharedMultiArray2;

template<typename T> class EnumeratorT;
template<typename T> class ListEnumeratorT;

template<typename T> class Collection;
template<typename T> class List;

class EventObservableBase;
class EventObserverBase;
template<typename... Args> class EventObservable;
template<typename... Args> class EventObserver;
template <typename... Args> class EventObservableView;

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
template<typename T,int Size> class NumVector;
template<typename T,int RowSize,int ColumnSize = RowSize> class NumMatrix;
using RealN2 = NumVector<Real,2>;
using RealN3 = NumVector<Real,3>;
using RealN2x2 = NumMatrix<Real,2>;
using RealN3x3 = NumMatrix<Real,3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
using MemoryView ARCANE_DEPRECATED_REASON("Use 'ConstMemoryView' instead") = ConstMemoryView;

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

class ForLoopRange;
template<int RankValue, typename IndexType_ = Int32> class SimpleForLoopRanges;
template<int RankValue, typename IndexType_ = Int32> class ComplexForLoopRanges;

// Pour compatibilité avec l'existant (à supprimer après la version 3.8)
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
//! Tableau dynamique à une dimension d'entiers 8 bits
typedef Array<Int8> Int8Array;
//! Tableau dynamique à une dimension d'entiers
typedef Array<Integer> IntegerArray;
//! Tableau dynamique à une dimension de réels
typedef Array<Real> RealArray;
//! Tableau dynamique à une dimension de 'bfloat16'
typedef Array<BFloat16> BFloat16Array;
//! Tableau dynamique à une dimension de 'float16'
typedef Array<Float16> Float16Array;
//! Tableau dynamique à une dimension de 'float'
typedef Array<Float32> Float32Array;
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
//! Tableau dynamique 2D d'entiers 8 bits
typedef Array2<Int8> Int8Array2;
//! Tableau dynamique 2D d'entiers
typedef Array2<Integer> IntegerArray2;
//! Tableau dynamique 2D de réels
typedef Array2<Real> RealArray2;
//! Tableau dynamique 2D de 'bfloat16'
typedef Array2<BFloat16> BFloat16Array2;
//! Tableau dynamique 2D de 'float16'
typedef Array2<Float16> Float16Array2;
//! Tableau dynamique 2D de 'float'
typedef Array2<Float32> Float32Array2;
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
//! Tableau dynamique 2D d'entiers 16 bits
typedef SharedArray2<Int16> Int16SharedArray2;
//! Tableau dynamique 2D d'entiers 8 bits
typedef SharedArray2<Int8> Int8SharedArray2;
//! Tableau dynamique 2D d'entiers
typedef SharedArray2<Integer> IntegerSharedArray2;
//! Tableau dynamique 2D de réels
typedef SharedArray2<Real> RealSharedArray2;
//! Tableau dynamique 2D de 'bfloat16'
typedef SharedArray2<BFloat16> BFloat16SharedArray2;
//! Tableau dynamique 2D de 'float16'
typedef SharedArray2<Float16> Float16SharedArray2;
//! Tableau dynamique 2D de 'float'
typedef SharedArray2<Float32> Float32SharedArray2;
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
//! Tableau dynamique à une dimension d'entiers 16 bits
typedef UniqueArray<Int8> Int8UniqueArray;
//! Tableau dynamique à une dimension d'entiers
typedef UniqueArray<Integer> IntegerUniqueArray;
//! Tableau dynamique à une dimension de réels
typedef UniqueArray<Real> RealUniqueArray;
//! Tableau dynamique à une dimension de 'bfloat16'
typedef UniqueArray<BFloat16> BFloat16UniqueArray;
//! Tableau dynamique à une dimension de 'float16'
typedef UniqueArray<Float16> Float16UniqueArray;
//! Tableau dynamique à une dimension de 'float'
typedef UniqueArray<Float32> Float32UniqueArray;
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
//! Tableau dynamique à une dimension d'entiers 16 bits
typedef SharedArray<Int8> Int8SharedArray;
//! Tableau dynamique à une dimension d'entiers
typedef SharedArray<Integer> IntegerSharedArray;
//! Tableau dynamique à une dimension de réels
typedef SharedArray<Real> RealSharedArray;
//! Tableau dynamique à une dimension de 'bfloat16'
typedef SharedArray<BFloat16> BFloat16SharedArray;
//! Tableau dynamique à une dimension de 'float16'
typedef SharedArray<Float16> Float16SharedArray;
//! Tableau dynamique à une dimension de 'float'
typedef SharedArray<Float32> Float32SharedArray;
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

//! Vue en lecture seule d'un tableau à une dimension de pointeurs
typedef Span<const Pointer> PointerConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de caractères
typedef Span<const std::byte> ByteConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de caractères unicode
typedef Span<const UChar> UCharConstSpan;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 64 bits
typedef Span<const Int64> Int64ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 32 bits
typedef Span<const Int32> Int32ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 16 bits
typedef Span<const Int16> Int16ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension d'entiers
typedef Span<const Integer> IntegerConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de réels
typedef Span<const Real> RealConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de booléens
typedef Span<const bool> BoolConstSpan;
//! Vue en lecture seule d'un tableau à une dimension d'entiers
typedef Span<const Integer> IntegerConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de Real2
typedef Span<const Real2> Real2ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de Real3
typedef Span<const Real3> Real3ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de Real2x2
typedef Span<const Real2x2> Real2x2ConstSpan;
//! Vue en lecture seule d'un tableau à une dimension de Real3x3
typedef Span<const Real3x3> Real3x3ConstSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Vue d'un tableau à une dimension de pointeurs
using PointerSmallSpan = SmallSpan<Pointer>;
//! Vue d'un tableau à une dimension de caractères
using ByteSmallSpan = SmallSpan<std::byte>;
//! Vue d'un tableau à une dimension de caractères unicode
using UCharSmallSpan = SmallSpan<UChar>;
//! Vue d'un tableau à une dimension d'entiers 64 bits
using Int64SmallSpan = SmallSpan<Int64>;
//! Vue d'un tableau à une dimension d'entiers 32 bits
using Int32SmallSpan = SmallSpan<Int32>;
//! Vue d'un tableau à une dimension d'entiers 16 bits
using Int16SmallSpan = SmallSpan<Int16>;
//! Vue d'un tableau à une dimension d'entiers
using IntegerSmallSpan = SmallSpan<Integer>;
//! Vue d'un tableau à une dimension de réels
using RealSmallSpan = SmallSpan<Real>;
//! Vue d'un tableau à une dimension de booléens
using BoolSmallSpan = SmallSpan<bool>;
//! Vue d'un tableau à une dimension d'entiers
using IntegerSmallSpan = SmallSpan<Integer>;
//! Vue d'un tableau à une dimension de Real2
using Real2SmallSpan = SmallSpan<Real2>;
//! Vue d'un tableau à une dimension de Real3
using Real3SmallSpan = SmallSpan<Real3>;
//! Vue d'un tableau à une dimension de Real2x2
using Real2x2SmallSpan = SmallSpan<Real2x2>;
//! Vue d'un tableau à une dimension de Real3x3
using Real3x3SmallSpan = SmallSpan<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Vue en lecture seule d'un tableau à une dimension de pointeurs
using PointerConstSmallSpan = SmallSpan<const Pointer>;
//! Vue en lecture seule d'un tableau à une dimension de caractères
using ByteConstSmallSpan = SmallSpan<const std::byte>;
//! Vue en lecture seule d'un tableau à une dimension de caractères unicode
using UCharConstSmallSpan = SmallSpan<const UChar>;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 64 bits
using Int64ConstSmallSpan = SmallSpan<const Int64>;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 32 bits
using Int32ConstSmallSpan = SmallSpan<const Int32>;
//! Vue en lecture seule d'un tableau à une dimension d'entiers 16 bits
using Int16ConstSmallSpan = SmallSpan<const Int16>;
//! Vue en lecture seule d'un tableau à une dimension d'entiers
using IntegerConstSmallSpan = SmallSpan<const Integer>;
//! Vue en lecture seule d'un tableau à une dimension de réels
using RealConstSmallSpan = SmallSpan<const Real>;
//! Vue en lecture seule d'un tableau à une dimension de booléens
using BoolConstSmallSpan = SmallSpan<const bool>;
//! Vue en lecture seule d'un tableau à une dimension d'entiers
using IntegerConstSmallSpan = SmallSpan<const Integer>;
//! Vue en lecture seule d'un tableau à une dimension de Real2
using Real2ConstSmallSpan = SmallSpan<const Real2>;
//! Vue en lecture seule d'un tableau à une dimension de Real3
using Real3ConstSmallSpan = SmallSpan<const Real3>;
//! Vue en lecture seule d'un tableau à une dimension de Real2x2
using Real2x2ConstSmallSpan = SmallSpan<const Real2x2>;
//! Vue en lecture seule d'un tableau à une dimension de Real3x3
using Real3x3ConstSmallSpan = SmallSpan<const Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
