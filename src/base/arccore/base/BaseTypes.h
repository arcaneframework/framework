/*---------------------------------------------------------------------------*/
/* BaseTypes.h                                                 (C) 2000-2018 */
/*                                                                           */
/* Définition des types de la composante 'base' de Arccore.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASETYPES_H
#define ARCCORE_BASE_BASETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

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

template<typename T> class IterT;
template<typename T> class ConstIterT;

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

template<typename T> class ConstArrayView;
template<typename T> class ArrayView;
template<typename T> class ConstArray2View;
template<typename T> class Array2View;
template<typename T> class ConstArray3View;
template<typename T> class Array3View;
template<typename T> class ConstArray4View;
template<typename T> class Array4View;
template<class DataType> class CoreArray;
template<typename T> class LargeArrayView;
template<typename T> class ConstLargeArrayView;

class StringImpl;
class String;
class StringBuilder;
class StringFormatterArg;

class TraceInfo;
class StackTrace;
class Exception;
class StackFrame;
class FixedStackFrameArray;
class IStackTraceService;

class ArgumentException;
class IndexOutOfRangeException;
class FatalErrorException;
class NotSupportedException;
class NotImplementedException;
class TimeoutException;

class IFunctor;
template<typename ArgType>
class IFunctorWithArgumentT;
template<typename ReturnType,typename Arg1,typename Arg2>
class IFunctorWithArgAndReturn2;
template<typename T>
class FunctorT;
template<typename ClassType,typename ArgType>
class FunctorWithArgumentT;
template<typename ArgType>
class StdFunctorWithArgumentT;

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
typedef LargeArrayView<Pointer> PointerLargeArrayView;
//! Equivalent C d'un tableau à une dimension de caractères
typedef LargeArrayView<Byte> ByteLargeArrayView;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef LargeArrayView<UChar> UCharLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef LargeArrayView<Int64> Int64LargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef LargeArrayView<Int32> Int32LargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef LargeArrayView<Int16> Int16LargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef LargeArrayView<Integer> IntegerLargeArrayView;
//! Equivalent C d'un tableau à une dimension de réels
typedef LargeArrayView<Real> RealLargeArrayView;
//! Equivalent C d'un tableau à une dimension de booléens
typedef LargeArrayView<bool> BoolLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef LargeArrayView<Integer> IntegerLargeArrayView;

//! Equivalent C d'un tableau à une dimension de pointeurs
typedef ConstLargeArrayView<Pointer> PointerConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension de caractères
typedef ConstLargeArrayView<Byte> ByteConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension de caractères unicode
typedef ConstLargeArrayView<UChar> UCharConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 64 bits
typedef ConstLargeArrayView<Int64> Int64ConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 32 bits
typedef ConstLargeArrayView<Int32> Int32ConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers 16 bits
typedef ConstLargeArrayView<Int16> Int16ConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ConstLargeArrayView<Integer> IntegerConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension de réels
typedef ConstLargeArrayView<Real> RealConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension de booléens
typedef ConstLargeArrayView<bool> BoolConstLargeArrayView;
//! Equivalent C d'un tableau à une dimension d'entiers
typedef ConstLargeArrayView<Integer> IntegerConstLargeArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
