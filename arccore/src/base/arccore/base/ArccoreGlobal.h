﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.h                                             (C) 2000-2020 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARCCOREGLOBAL_H
#define ARCCORE_BASE_ARCCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <cstdint>

#include "arccore/arccore_config.h"

#ifdef ARCANE_VALID_TARGET
#  undef ARCANE_VALID_TARGET
#endif

// Determine le type de l'os.
#if defined(__linux)
#  define ARCCORE_OS_LINUX
#elif defined(__APPLE__)
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
 - ARCANE_DEVICE_CODE: indique une partie de code compilée uniquement sur le device
 - ARCANE_HOST_DEVICE: indique que la méthode/variable est accessible à la fois
   sur le device et l'hôte
 - ARCANE_DEVICE: indique que la méthode/variable est accessible uniquement sur
   le device.
*/

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define ARCCORE_DEVICE_CODE
#if defined(__CUDA_ARCH__)
#define ARCCORE_DEVICE_TARGET_CUDA
#if defined(__HIPSYCL__)
// Nécessaire pour assert() par exemple dans arccoreCheckAt()
// TODO: regarder si cela est aussi nécessaire pour AMD HIP.
#include <cassert>
#endif
#endif
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

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Définition des types Arccore Int16, Int32 et Int64.
 *
 * Ces types sont définis lors de la configuration dans le fichier
 * 'arccore_config.h'.
 */
typedef std::int16_t Int16;
typedef std::int32_t Int32;
typedef std::int64_t Int64;
typedef std::uint32_t UInt32;
typedef std::uint64_t UInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type représentant un pointeur.
 *
 * Il doit être utilisé partout ou un objet de type pointeur quelconque est attendu.
 */
typedef void* Pointer;

#ifdef ARCCORE_REAL_USE_APFLOAT
#  define ARCCORE_REAL(val) (Real(#val,1000))
#  define ARCCORE_REAL_NOT_BUILTIN
typedef apfloat Real;
typedef apfloat APReal;
#else
#  ifdef ARCCORE_REAL_LONG
#    define ARCCORE_REAL(val) val##L
/*!
 * \brief Type représentant un réel.
 *
 * Il doit être utilisé partout ou un objet de type réel est attendu.
 */
typedef long double Real;
#  else
#    define ARCCORE_REAL(val) val
#    define ARCCORE_REAL_IS_DOUBLE
/*!
 * \brief Type représentant un réel.
 *
 * Il doit être utilisé partout ou un objet de type réel est attendu.
 */
typedef double Real;
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
typedef Int32 Short;
typedef Int64 Integer;
typedef Int64 Integer;
#else
#  define ARCCORE_INTEGER_MAX ARCCORE_INT32_MAX
typedef Int32 Short;
typedef Int32 Integer;
typedef Int32 Integer;
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
 * la macro sont utilisés formatter un message d'erreur via la
 * méthode String::format().
 */
#define ARCCORE_THROW(exception_class,...)                           \
  throw exception_class (A_FUNCINFO,Arccore::String::format(__VA_ARGS__))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro envoyant une exception FatalErrorException.
 *
 * Les arguments de la macro sont utilisés formatter un message
 * d'erreur via la méthode String::format().
 */
#define ARCCORE_FATAL(...)\
  throw Arccore::FatalErrorException(A_FUNCINFO,Arccore::String::format(__VA_ARGS__))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signalue l'utilisation d'un pointeur nul.
 *
 * Signale une tentative d'utilisation d'un pointeur nul.
 * Affiche un message, appelle arcaneDebugPause() et lance une exception
 * de type FatalErrorException.
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreNullPointerError ARCCORE_NORETURN ();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signale une erreur de débordement.
 *
 * Signale un débordement de tableau. Affiche un message et appelle
 * arcaneDebugPause().
 *
 * \param i indice invalide
 * \param max_size nombre d'éléments du tableau
 */
extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError ARCCORE_NORETURN (Int64 i,Int64 max_size);

/*!
 * \brief Vérifie un éventuel débordement de tableau.
 */
inline ARCCORE_HOST_DEVICE void
arccoreCheckAt(Int64 i,Int64 max_size)
{
#ifndef ARCCORE_DEVICE_CODE
  if (i<0 || i>=max_size)
    arccoreRangeError(i,max_size);
#else
  // Code pour le device.
  // assert() est disponible pour CUDA.
  // TODO: regarder si une fonction similaire existe pour HIP
#ifdef ARCCORE_DEVICE_TARGET_CUDA
  assert(i>=0 && i<max_size);
#else
  ARCCORE_UNUSED(i);
  ARCCORE_UNUSED(max_size);
#endif
#endif
}

#if defined(ARCCORE_CHECK) || defined(ARCCORE_DEBUG)
#define ARCCORE_CHECK_AT(a,b) ::Arccore::arccoreCheckAt((a),(b))
#else
#define ARCCORE_CHECK_AT(a,b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_CAST_SMALL_SIZE(a) ((Integer)(a))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Macros utilisées pour le débug.
 */
#ifdef ARCCORE_DEBUG_ASSERT
extern "C++" ARCCORE_BASE_EXPORT void _doAssert(const char*,const char*,const char*,int);
template<typename T> inline T*
_checkPointer(T* t,const char* file,const char* func,int line)
{
  if (!t){
    _doAssert("ARCCORE_ASSERT",file,func,line);
    arccorePrintf("Bad Pointer");
  }
  return t;
}
#  ifdef __GNUG__
#    define ARCCORE_D_WHERE(a)  Arccore::_doAssert(a,__FILE__,__PRETTY_FUNCTION__,__LINE__)
#    define ARCCORE_DCHECK_POINTER(a) Arccore::_checkPointer((a),__FILE__,__PRETTY_FUNCTION__,__LINE__);
#  else
#    define ARCCORE_D_WHERE(a)  Arccore::_doAssert(a,__FILE__,"(NoInfo)",__LINE__)
#    define ARCCORE_DCHECK_POINTER(a) Arccore::_checkPointer((a),__FILE__,"(NoInfo"),__LINE__);
#  endif
#  define ARCCORE_CHECK_PTR(a) \
   {if (!(a)){Arccore::arccorePrintf("Null value");ARCCORE_D_WHERE("ARCCORE_ASSERT");}}

#  define ARCCORE_ASSERT(a,b) \
  {if (!(a)){ Arccore::arccorePrintf("Assertion '%s' fails:",#a); Arccore::arccorePrintf b; ARCCORE_D_WHERE("ARCCORE_ASSERT");}}
#  define ARCCORE_WARNING(a) \
   { Arccore::arccorePrintf a; ARCCORE_D_WHERE("ARCCORE_WARNING"); }
#else
#  define ARCCORE_CHECK_PTR(a)
#  define ARCCORE_ASSERT(a,b)
#  define ARCCORE_WARNING(a)
#  define ARCCORE_DCHECK_POINTER(a) (a);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Défitions des types de base.
class String;
class StringView;
class StringFormatterArg;
class StringBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
