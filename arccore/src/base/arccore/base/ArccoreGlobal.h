// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.h                                             (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARCCOREGLOBAL_H
#define ARCCORE_BASE_ARCCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <cstdint>

#include "arccore/arccore_config.h"

#ifdef ARCCORE_VALID_TARGET
#  undef ARCCORE_VALID_TARGET
#endif

// Determine le type de l'os.
#if defined(__linux)
#  define ARCCORE_OS_LINUX
#elif defined(__APPLE__) && defined(__MACH__)
#  define ARCCORE_OS_MACOS
#elif defined(_AIX)
#  define ARCCORE_OS_AIX
#elif defined(__WIN32__) || defined(__NT__) || defined(WIN32) || defined(_WIN32) || defined(WIN32) || defined(_WINDOWS)
#  define ARCCORE_OS_WIN32
#elif defined(__CYGWIN__)
#  define ARCCORE_OS_CYGWIN
#endif

#ifdef ARCCORE_OS_WIN32
#  define ARCCORE_VALID_TARGET
#  define ARCCORE_EXPORT     __declspec(dllexport)
#  define ARCCORE_IMPORT     __declspec(dllimport)

/* Supprime certains avertissements du compilateur Microsoft */
#  ifdef _MSC_VER
#    pragma warning(disable: 4251) // class 'A' needs to have dll interface for to be used by clients of class 'B'.
#    pragma warning(disable: 4275) // non - DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'
#    pragma warning(disable: 4800) // 'type' : forcing value to bool 'true' or 'false' (performance warning)
#    pragma warning(disable: 4355) // 'this' : used in base member initializer list
#  endif

#endif

// Sous Unix, indique que par défaut les symboles de chaque .so sont cachés.
// Il faut alors explicitement marquer les
// symboles qu'on souhaite exporter, comme sous windows.
// La seule différence est que pour gcc avec les instantiations explicites
// de template, il faut spécifier l'export lors de l'instantiation
// explicite alors que sous windows c'est dans la classe.
#ifndef ARCCORE_OS_WIN32
#  define ARCCORE_EXPORT __attribute__ ((visibility("default")))
#  define ARCCORE_IMPORT __attribute__ ((visibility("default")))
#  define ARCCORE_TEMPLATE_EXPORT ARCCORE_EXPORT
#endif

#ifdef ARCCORE_OS_CYGWIN
#  define ARCCORE_VALID_TARGET
#endif

#ifdef ARCCORE_OS_LINUX
#  define ARCCORE_VALID_TARGET
#endif

#ifdef ARCCORE_OS_MACOS
#  define ARCCORE_VALID_TARGET
#endif

#ifndef ARCCORE_VALID_TARGET
#error "This target is not supported"
#endif

#ifndef ARCCORE_EXPORT
#define ARCCORE_EXPORT
#endif

#ifndef ARCCORE_IMPORT
#define ARCCORE_IMPORT
#endif

#ifndef ARCCORE_TEMPLATE_EXPORT
#define ARCCORE_TEMPLATE_EXPORT
#endif

#ifndef ARCCORE_RESTRICT
#define ARCCORE_RESTRICT
#endif

#define ARCCORE_STD std

//Tag var as a voluntary unused variable.
//Works with any compiler but might be improved by using attribute.
#define ARCCORE_UNUSED(var) do { (void)(var) ; } while(false)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef DOXYGEN_DOC
typedef ARCCORE_TYPE_INT16 Int16;
typedef ARCCORE_TYPE_INT32 Int32;
typedef ARCCORE_TYPE_INT64 Int64;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Macros pour le support de la programmation hétérogène (CPU/GPU)
 - ARCCORE_DEVICE_CODE: indique une partie de code compilée uniquement sur le device
 - ARCCORE_HOST_DEVICE: indique que la méthode/variable est accessible à la fois
   sur le device et l'hôte
 - ARCCORE_DEVICE: indique que la méthode/variable est accessible uniquement sur
   le device.
*/

#if defined(__SYCL_DEVICE_ONLY__)
#  define ARCCORE_DEVICE_CODE
#  define ARCCORE_DEVICE_TARGET_SYCL
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#  define ARCCORE_DEVICE_CODE
#  if defined(__HIP_DEVICE_COMPILE__)
#    define ARCCORE_DEVICE_TARGET_HIP
#  endif
#  if defined(__CUDA_ARCH__)
#    define ARCCORE_DEVICE_TARGET_CUDA
// Nécessaire pour assert() par exemple dans arccoreCheckAt()
// TODO: regarder si cela est aussi nécessaire pour AMD HIP.
#include <cassert>
#  endif
#endif

#if defined(__CUDACC__) || defined(__HIP__)
#define ARCCORE_HOST_DEVICE __host__ __device__
#define ARCCORE_DEVICE __device__
#endif


#ifndef ARCCORE_HOST_DEVICE
#define ARCCORE_HOST_DEVICE
#endif

#ifndef ARCCORE_DEVICE
#define ARCCORE_DEVICE
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_base)
#define ARCCORE_BASE_EXPORT ARCCORE_EXPORT
#define ARCCORE_BASE_EXTERN_TPL
#else
#define ARCCORE_BASE_EXPORT ARCCORE_IMPORT
#define ARCCORE_BASE_EXTERN_TPL extern
#endif

#ifdef ARCCORE_REAL_USE_APFLOAT
#  include <apfloat.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Définition des types Arccore Int16, Int32 et Int64.
 */
//! Type entier signé sur 8 bits
using Int8 = std::int8_t;
//! Type entier signé sur 16 bits
using Int16 = std::int16_t;
//! Type entier signé sur 32 bits
using Int32 = std::int32_t;
//! Type entier signé sur 64 bits
using Int64 = std::int64_t;
//! Type entier non signé sur 32 bits
using UInt32 = std::uint32_t;
//! Type entier non signé sur 64 bits
using UInt64 = std::uint64_t;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type représentant un pointeur.
 *
 * Il doit être utilisé partout ou un objet de type pointeur quelconque est attendu.
 */
using Pointer = void*;

#ifdef ARCCORE_REAL_USE_APFLOAT
#  define ARCCORE_REAL(val) (Real(#val,1000))
#  define ARCCORE_REAL_NOT_BUILTIN
using Real = apfloat;
using APReal = apfloat;
#else
#  ifdef ARCCORE_REAL_LONG
#    define ARCCORE_REAL(val) val##L
/*!
 * \brief Type représentant un réel.
 *
 * Il doit être utilisé partout ou un objet de type réel est attendu.
 */
using long double Real;
#  else
#    define ARCCORE_REAL(val) val
#    define ARCCORE_REAL_IS_DOUBLE
/*!
 * \brief Type représentant un réel.
 *
 * Il doit être utilisé partout ou un objet de type réel est attendu.
 */
using Real = double;
#  endif
//! Emulation de réel en précision arbitraire.
class APReal
{
 public:
  Real v[4];
};
#endif

#ifdef ARCCORE_64BIT
#  define ARCCORE_INTEGER_MAX ARCCORE_INT64_MAX
using Short = Int32;
using Integer = Int64;
#else
#  define ARCCORE_INTEGER_MAX ARCCORE_INT32_MAX
using Short = Int32;
using Integer = Int32;
#endif

/*!
 * \def ARCCORE_INTEGER_MAX
 * \brief Macro indiquant la valeur maximal que peut prendre le type #Integer
 */


/*!
 * \typedef Int64
 * \brief Type entier signé sur 64 bits.
 */
/*!
 * \typedef Int32
 * \brief Type entier signé sur 32 bits.
 */
/*!
 * \typedef Int16
 * \brief Type entier signé sur 16 bits.
 */
/*!
 * \typedef Integer
 * \brief Type représentant un entier
 *
 * Si la macro ARCCORE_64BIT est définie, le type Integer correspond à un
 * entier Int64, sinon à un entier Int32.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Brain Float16
class BFloat16;

//! Float 16 bit
class Float16;

//! Type flottant IEEE-753 simple précision
using Float32 = float;

//! Float 128 bit
class Float128;

//! Integer 128 bit
class Int128;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Structure équivalente à la valeur booléenne \a vrai
 */
struct TrueType  {};
/*!
  \internal
  \brief Structure équivalente à la valeur booléenne \a vrai
*/
struct FalseType {};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#  define ARCCORE_DEPRECATED __attribute__ ((deprecated))
#endif

#ifdef _MSC_VER
#  if _MSC_VER >= 1300
#    define ARCCORE_DEPRECATED __declspec(deprecated)
#  endif
#endif

#define ARCCORE_DEPRECATED_2017 ARCCORE_DEPRECATED
#define ARCCORE_DEPRECATED_2018 ARCCORE_DEPRECATED
#define ARCCORE_DEPRECATED_2019(reason) [[deprecated(reason)]]
#define ARCCORE_DEPRECATED_2020(reason) [[deprecated(reason)]]
#define ARCCORE_DEPRECATED_REASON(reason) [[deprecated(reason)]]

// Définir cette macro si on souhaite supprimer de la compilation les
// méthodes et types obsolètes.
#define ARCCORE_NO_DEPRECATED

#ifndef ARCCORE_DEPRECATED
#  define ARCCORE_DEPRECATED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Macros de compatibilités avec les différents standards du C++.
// Maintenant (en 2021) tous les compilateurs avec lesquels Arccore compile
// ont le support du C++17 donc la plupart de ces macros ne sont plus utiles.
// On les garde uniquement pour compatibilité avec le code existant.

// La macro ARCCORE_NORETURN utilise l'attribut [[noreturn]] du C++11 pour
// indiquer qu'une fonction ne retourne pas.
#define ARCCORE_NORETURN [[noreturn]]

//! Macro permettant de spécifier le mot-clé 'constexpr' du C++11
#define ARCCORE_CONSTEXPR constexpr

// Macro pour indiquer qu'on ne lance pas d'exceptions.
#define ARCCORE_NOEXCEPT noexcept

// Macros pour indiquer qu'on lance pas d'exceptions.
#define ARCCORE_NOEXCEPT_FALSE noexcept(false)

// Support pour operator[](a,b,...)
#ifdef __cpp_multidimensional_subscript
#define ARCCORE_HAS_MULTI_SUBSCRIPT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Macros pour l'attribut [[no_unique_address]]
// Avec VS2022, cet attribut n'est pas pris en compte et il faut
// utiliser [[msvc::no_unique_address]]
#ifdef _MSC_VER
#define ARCCORE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define ARCCORE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Support pour l'alignement.
// le C++11 utilise le mot clé alignas pour spécifier l'alignement.
// Cela fonctionne avec GCC 4.9+ et Visual Studio 2015. Cela ne fonctionne
// pas avec Visual Studio 2013. Donc pour Visual Studio on utilise dans tous
// les cas __declspec qui fonctionne toujours. Sous Linux, __attribute__ fonctionne
// aussi toujours donc on utilise cela. A noter que les structures Simd ont besoin
// de l'attribut 'packed' qui n'existe que avec GCC et Intel. Il ne semble pas y avoir
// d'équivalent avec MSVC.
#ifdef _MSC_VER
//! Macro pour garantir le compactage et l'alignement d'une classe sur \a value octets
#  define ARCCORE_ALIGNAS(value) __declspec(align(value))
//! Macro pour garantir l'alignement d'une classe sur \a value octets
#  define ARCCORE_ALIGNAS_PACKED(value) __declspec(align(value))
#else
//! Macro pour garantir le compactage et l'alignement d'une classe sur \a value octets
#  define ARCCORE_ALIGNAS_PACKED(value) __attribute__ ((aligned (value),packed))
//! Macro pour garantir l'alignement d'une classe sur \a value octets
#  define ARCCORE_ALIGNAS(value) __attribute__ ((aligned (value)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_CHECK) || defined(ARCCORE_DEBUG)
#  ifndef ARCCORE_DEBUG_ASSERT
#    define ARCCORE_DEBUG_ASSERT
#  endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vrai si on est en mode vérification.
 *
 * Ce mode est actif si la macro ARCCORE_CHECK est définie
 * ou si la méthode arccoreSetCheck() a été positionnée a vrai.
 */
extern "C++" ARCCORE_BASE_EXPORT 
bool arccoreIsCheck();

/*!
 * \brief Active ou désactive le mode vérification.
 *
 * Le mode vérification est toujours actif si la macro ARCCORE_CHECK est définie.
 * Sinon, il est possible de l'activer avec cette méthode. Cela permet
 * d'activer certains tests même en mode optimisé.
 */
extern "C++" ARCCORE_BASE_EXPORT 
void arccoreSetCheck(bool v);

/*!
 * \brief Vrai si la macro ARCCORE_DEBUG est définie
 */
extern "C++" ARCCORE_BASE_EXPORT
bool arccoreIsDebug();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Encapsulation de la fonction C printf
extern "C++" ARCCORE_BASE_EXPORT void
arccorePrintf(const char*,...);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Passe en mode pause ou lance une erreur fatale.
 *
 * Si arccoreSetPauseOnError() est appelé avec l'argument \a true,
 * met le programme en pause
 * pour éventuellement connecter un débugger dessus.
 *
 * Sinon, lance une exception FatalErrorException avec le message
 * \a msg comme argument.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreDebugPause(const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique si on l'appel à arccoreDebugPause() effectue une pause.
 *
 * \sa arccoreDebugPause()
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreSetPauseOnError(bool v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour envoyer une exception avec formattage.
 *
 * \a exception_class est le type de l'exception. Les arguments suivants de
 * la macro sont utilisés pour formatter un message d'erreur via la
 * méthode String::format().
 */
#define ARCCORE_THROW(exception_class,...) \
  throw exception_class (A_FUNCINFO,Arccore::String::format(__VA_ARGS__))

/*!
 * \brief Macro pour envoyer une exception avec formattage si \a cond est vrai.
 *
 * \a exception_class est le type de l'exception. Les arguments suivants de
 * la macro sont utilisés pour formatter un message d'erreur via la
 * méthode String::format().
 *
 * \sa ARCCORE_THROW
 */
#define ARCCORE_THROW_IF(cond, exception_class, ...) \
  if (cond) [[unlikely]] \
    ARCCORE_THROW(exception_class,__VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro envoyant une exception FatalErrorException.
 *
 * Les arguments de la macro sont utilisés pour formatter un message
 * d'erreur via la méthode String::format().
 */
#define ARCCORE_FATAL(...)\
  ARCCORE_THROW(::Arccore::FatalErrorException,__VA_ARGS__)

/*!
 * \brief Macro envoyant une exception FatalErrorException si \a cond est vrai
 *
 * Les arguments de la macro sont utilisés pour formatter un message
 * d'erreur via la méthode String::format().
 *
 * \sa ARCCORE_FATAL
 */
#define ARCCORE_FATAL_IF(cond, ...) \
  ARCCORE_THROW_IF(cond, ::Arccore::FatalErrorException,__VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signalue l'utilisation d'un pointeur nul.
 *
 * Signale une tentative d'utilisation d'un pointeur nul.
 * Affiche un message, appelle arccoreDebugPause() et lance une exception
 * de type FatalErrorException.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreNullPointerError ARCCORE_NORETURN ();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signale qu'une valeur n'est pas dans l'intervalle souhaité.
 *
 * Indique que l'assertion `min_value_inclusive <= i < max_value_exclusive`
 * est fausse.
 * Appelle arccoreDebugPause() puis lève une exception de type
 * IndexOutOfRangeException.
 *
 * \param i valeur invalide.
 * \param min_value_inclusive valeur minimale inclusive autorisée.
 * \param max_value_exclusive valeur maximale exclusive autorisée.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError ARCCORE_NORETURN (Int64 i,Int64 min_value_inclusive,
                                    Int64 max_value_exclusive);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signale qu'une valeur n'est pas dans l'intervalle souhaité.
 *
 * Indique que l'assertion `0 <= i < max_value est fausse`.
 * Lance une execption IndexOutOfRangeException.
 *
 * \param i indice invalide
 * \param max_size nombre d'éléments du tableau
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError ARCCORE_NORETURN (Int64 i,Int64 max_size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que `min_value_inclusive <= i < max_value_exclusive`.
 *
 * Si ce n'est pas le cas, appelle arccoreRangeError() pour lancer une
 * exception.
 */
inline ARCCORE_HOST_DEVICE void
arccoreCheckRange(Int64 i,Int64 min_value_inclusive,Int64 max_value_exclusive)
{
  if (i>=min_value_inclusive && i<max_value_exclusive)
    return;
#ifndef ARCCORE_DEVICE_CODE
  arccoreRangeError(i,min_value_inclusive,max_value_exclusive);
#elif defined(ARCCORE_DEVICE_TARGET_CUDA)
  // Code pour le device.
  // assert() est disponible pour CUDA.
  // TODO: regarder si une fonction similaire existe pour HIP
  assert(false);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie un éventuel débordement de tableau.
 *
 * Appelle arccoreCheckRange(i,0,max_size).
 */
inline ARCCORE_HOST_DEVICE void
arccoreCheckAt(Int64 i,Int64 max_size)
{
  arccoreCheckRange(i,0,max_size);
}

#if defined(ARCCORE_CHECK) || defined(ARCCORE_DEBUG)
#define ARCCORE_CHECK_AT(a,b) ::Arccore::arccoreCheckAt((a),(b))
#define ARCCORE_CHECK_RANGE(a,b,c) ::Arccore::arccoreCheckRange((a),(b),(c))
#else
#define ARCCORE_CHECK_AT(a,b)
#define ARCCORE_CHECK_RANGE(a,b,c)
#endif

#define ARCCORE_CHECK_AT2(a0,a1,b0,b1) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1)
#define ARCCORE_CHECK_AT3(a0,a1,a2,b0,b1,b2) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1); ARCCORE_CHECK_AT(a2,b2)
#define ARCCORE_CHECK_AT4(a0,a1,a2,a3,b0,b1,b2,b3) \
  ARCCORE_CHECK_AT(a0,b0); ARCCORE_CHECK_AT(a1,b1); ARCCORE_CHECK_AT(a2,b2); ARCCORE_CHECK_AT(a3,b3)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_CAST_SMALL_SIZE(a) ((Integer)(a))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
_doAssert(const char*,const char*,const char*,int);
template<typename T> inline T*
_checkPointer(const T* t,const char* file,const char* func,int line)
{
  if (!t){
    _doAssert("ARCCORE_ASSERT",file,func,line);
    arccorePrintf("Bad Pointer");
  }
  return t;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Macro pour obtenir par le pré-processeur le nom de la fonction actuelle

#if defined(__GNUG__)
#  define ARCCORE_MACRO_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined( _MSC_VER)
#  define ARCCORE_MACRO_FUNCTION_NAME __FUNCTION__
#else
#  define ARCCORE_MACRO_FUNCTION_NAME __func__
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Macros utilisées pour le débug.
 */
#ifdef ARCCORE_DEBUG_ASSERT
#  define ARCCORE_D_WHERE(a) ::Arcane::_doAssert(a, __FILE__, ARCCORE_MACRO_FUNCTION_NAME, __LINE__)
#  define ARCCORE_DCHECK_POINTER(a) ::Arcane::_checkPointer((a), __FILE__, ARCCORE_MACRO_FUNCTION_NAME, __LINE__);
#  define ARCCORE_CHECK_PTR(a) \
  { \
    if (!(a)) { \
      ::Arcane::arccorePrintf("Null value"); \
      ARCCORE_D_WHERE("ARCCORE_ASSERT"); \
    } \
  }

#  define ARCCORE_ASSERT(a,b) \
  { \
    if (!(a)) { \
      ::Arcane::arccorePrintf("Assertion '%s' fails:", #a); \
      ::Arcane::arccorePrintf b; \
      ARCCORE_D_WHERE("ARCCORE_ASSERT"); \
    } \
  }
#  define ARCCORE_WARNING(a) \
  { \
    ::Arcane::arccorePrintf a; \
    ARCCORE_D_WHERE("ARCCORE_WARNING"); \
  }
#else
#  define ARCCORE_CHECK_PTR(a)
#  define ARCCORE_ASSERT(a,b)
#  define ARCCORE_WARNING(a)
#  define ARCCORE_DCHECK_POINTER(a) (a);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signalee l'utilisation d'un pointeur nul en envoyant une exception
 *
 * Signale une tentative d'utilisation d'un pointeur nul.
 * Lance une exception de type FatalErrorException.
 *
 * Dans l'exception, affiche \a text si non nul, sinon affiche \a ptr_name.
 *
 * Normalement cette méthode ne doit pas être appelée directement mais
 * via la macro ARCCORE_CHECK_POINTER.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowNullPointerError [[noreturn]] (const char* ptr_name,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 *
 * Si le pointeur est nul, appelle arccoreThrowNullPointerError().
 * Sinon, retourne le pointeur.
 */
inline void*
arccoreThrowIfNull(void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arccoreThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 *
 * Si le pointeur est nul, appelle arccoreThrowNullPointerError().
 * Sinon, retourne le pointeur.
 */
inline const void*
arccoreThrowIfNull(const void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arccoreThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro retournant le pointeur \a ptr s'il est non nul
 * ou lancant une exception s'il est nul.
 *
 * \sa arccoreThrowIfNull().
 */
#define ARCCORE_CHECK_POINTER(ptr) \
  arccoreThrowIfNull(ptr,#ptr,nullptr)

/*!
 * \brief Macro retournant le pointeur \a ptr s'il est non nul
 * ou lancant une exception s'il est nul.
 *
 * \sa arccoreThrowIfNull().
 */
#define ARCCORE_CHECK_POINTER2(ptr,text)\
  arccoreThrowIfNull(ptr,#ptr,text)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Défitions des types de base.
class String;
class StringView;
class StringFormatterArg;
class StringBuilder;
// Pas dans cette composante mais comme cette interface on la met ici
// pour compatibilité avec l'existant
class ITraceMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::ITraceMng;
using Arcane::String;
using Arcane::StringBuilder;
using Arcane::StringFormatterArg;
using Arcane::StringView;
using Arcane::UInt32;
using Arcane::UInt64;

using Arcane::APReal;
using Arcane::Integer;
using Arcane::Pointer;
using Arcane::Real;
using Arcane::Short;

//! Type 'Brain Float16'
using BFloat16 = Arcane::BFloat16;

//! Type 'Float16' (binary16)
using Float16 = Arcane::Float16;

//! Type flottant IEEE-753 simple précision (binary32)
using Float32 = float;

//! Type représentant un entier sur 8 bits
using Int8 = Arcane::Int8;

//! Type représentant un floattan sur 128 bits
using Float128 = Arcane::Float128;

//! Type représentant un entier sur 128 bits
using Int128 = Arcane::Int128;
using Int16 = Arcane::Int16;
using Int32 = Arcane::Int32;
using Int64 = Arcane::Int64;

using Arcane::arccoreCheckAt;
using Arcane::arccoreCheckRange;
using Arcane::arccoreDebugPause;
using Arcane::arccoreIsCheck;
using Arcane::arccoreIsDebug;
using Arcane::arccoreNullPointerError;
using Arcane::arccorePrintf;
using Arcane::arccoreRangeError;
using Arcane::arccoreSetCheck;
using Arcane::arccoreSetPauseOnError;
using Arcane::arccoreThrowIfNull;
using Arcane::arccoreThrowNullPointerError;

using Arcane::FalseType;
using Arcane::TrueType;
using Arcane::_doAssert;
using Arcane::_checkPointer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
