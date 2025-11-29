// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneGlobal.h                                              (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales de Arcane.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARCANEGLOBAL_H
#define ARCANE_UTILS_ARCANEGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

// Infos globales sur les options de compilation comme
// les threads, le mode debug, ...
#include "arcane_core_config.h"

#ifdef ARCCORE_OS_LINUX
#  define ARCANE_OS_LINUX
#  include <cstddef>
#endif

#ifdef ARCCORE_OS_WIN32
#  define ARCANE_OS_WIN32
#endif

#ifdef ARCCORE_OS_MACOS
#  define ARCANE_OS_MACOS
#endif

#define ARCANE_EXPORT ARCCORE_EXPORT
#define ARCANE_IMPORT ARCCORE_IMPORT
#define ARCANE_TEMPLATE_EXPORT ARCCORE_TEMPLATE_EXPORT
#define ARCANE_RESTRICT ARCCORE_RESTRICT

#define ARCANE_STD std

//Tag var as a voluntary unused variable.
//Works with any compiler but might be improved by using attribute.
#define ARCANE_UNUSED(var) ARCCORE_UNUSED(var)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_HAS_CUDA) && defined(__CUDACC__)
/*!
 * \brief Macro pour indiquer qu'on compile %Arcane avec le support
 * de CUDA et qu'on utilise le compilateur CUDA.
 */
#define ARCANE_COMPILING_CUDA
#endif
#if defined(ARCANE_HAS_HIP) && defined(__HIP__)
/*!
 * \brief Macro pour indiquer qu'on compile %Arcane avec le support
 * de HIP et qu'on utilise le compilateur HIP.
 */
#define ARCANE_COMPILING_HIP
#endif

#if defined(ARCANE_HAS_SYCL)
#  if defined(SYCL_LANGUAGE_VERSION) || defined(__ADAPTIVECPP__)
/*!
 * \brief Macro pour indiquer qu'on compile %Arcane avec le support
 * de SYCL et qu'on utilise le compilateur SYCL.
 */
#    define ARCANE_COMPILING_SYCL
#  endif
#endif

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
#define ARCANE_COMPILING_CUDA_OR_HIP
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: supprimer l'inclusion de <iosfwd> et les using.
// Pour l'instant (2022), on supprime ces inclusions uniquement pour Arcane.

#ifndef ARCANE_NO_USING_FOR_STREAM
#include <iosfwd>
using std::istream;
using std::ostream;
using std::ios;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::istringstream;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef DOXYGEN_DOC
typedef ARCANE_TYPE_INT16 Int16;
typedef ARCANE_TYPE_INT32 Int32;
typedef ARCANE_TYPE_INT64 Int64;
#endif

#define ARCANE_BEGIN_NAMESPACE  namespace Arcane {
#define ARCANE_END_NAMESPACE    }
#define NUMERICS_BEGIN_NAMESPACE  namespace Numerics {
#define NUMERICS_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_FULL
#define ARCANE_COMPONENT_arcane_utils
#define ARCANE_COMPONENT_arcane
#define ARCANE_COMPONENT_arcane_mesh
#define ARCANE_COMPONENT_arcane_std
#define ARCANE_COMPONENT_arcane_impl
#define ARCANE_COMPONENT_arcane_script
#endif

#if defined(ARCANE_COMPONENT_arcane) || defined(ARCANE_COMPONENT_arcane_core)
#define ARCANE_CORE_EXPORT ARCANE_EXPORT
#define ARCANE_EXPR_EXPORT ARCANE_EXPORT
#define ARCANE_DATATYPE_EXPORT ARCANE_EXPORT
#define ARCANE_CORE_EXTERN_TPL
#else
#define ARCANE_CORE_EXPORT ARCANE_IMPORT
#define ARCANE_EXPR_EXPORT ARCANE_IMPORT
#define ARCANE_DATATYPE_EXPORT ARCANE_IMPORT
#define ARCANE_CORE_EXTERN_TPL extern
#endif

#ifdef ARCANE_COMPONENT_arcane_utils
#define ARCANE_UTILS_EXPORT ARCANE_EXPORT
#define ARCANE_UTILS_EXTERN_TPL
#else
#define ARCANE_UTILS_EXPORT ARCANE_IMPORT
#define ARCANE_UTILS_EXTERN_TPL extern
#endif

#ifdef ARCANE_COMPONENT_arcane_impl
#define ARCANE_IMPL_EXPORT ARCANE_EXPORT
#else
#define ARCANE_IMPL_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_mesh
#define ARCANE_MESH_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MESH_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_std
#define ARCANE_STD_EXPORT ARCANE_EXPORT
#else
#define ARCANE_STD_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_script
#define ARCANE_SCRIPT_EXPORT ARCANE_EXPORT
#else
#define ARCANE_SCRIPT_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_solvers
#define ARCANE_SOLVERS_EXPORT ARCANE_EXPORT
#else
#define ARCANE_SOLVERS_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_geometry
#define ARCANE_GEOMETRY_EXPORT ARCANE_EXPORT
#else
#define ARCANE_GEOMETRY_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_thread
#define ARCANE_THREAD_EXPORT ARCANE_EXPORT
#else
#define ARCANE_THREAD_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_mpi
#define ARCANE_MPI_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MPI_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_COMPONENT_arcane_hyoda
#define ARCANE_HYODA_EXPORT ARCANE_EXPORT
#else
#define ARCANE_HYODA_EXPORT ARCANE_IMPORT
#endif

#ifdef ARCANE_REAL_USE_APFLOAT
#include <apfloat.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_HAS_LONG_LONG

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const double cgrEPSILON_DELTA = 1.0e-2;
const double cgrPI = 3.14159265358979323846;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_REAL(val) ARCCORE_REAL(val)

#ifdef ARCCORE_REAL_NOT_BUILTIN
#  define ARCANE_REAL_NOT_BUILTIN
#endif

#ifdef ARCCORE_REAL_LONG
#  define ARCANE_REAL_LONG
#endif

#ifdef ARCCORE_REAL_IS_DOUBLE
#  define ARCANE_REAL_IS_DOUBLE
#endif

/*!
 * \brief Type des entiers utilisés pour stocker les identifiants locaux
 * des entités.
 *
 * Les valeurs que peut prendre ce type indique combien d'entités
 * pourront être présentes sur un sous-domaine.
 */
using LocalIdType = Int32;

/*!
 * \brief Type des entiers utilisés pour stocker les identifiants uniques
 * (globaux) des entités.
 *
 * Les valeurs que peut prendre ce type indique combien d'entités
 * pourront être présentes sur le domaine initial.
 */
using UniqueIdType = Int64;

/*!
 * \def ARCANE_INTEGER_MAX
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
 * Si la macro ARCANE_64BIT est définie, le type Integer correspond à un
 * entier Int64, sinon à un entier Int32.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Encapsulation de la fonction C printf
extern "C++" ARCANE_UTILS_EXPORT void
arcanePrintf(const char*,...);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Passe en mode pause ou lance une erreur fatale.
 *
 * Si le code est compilé en mode \a debug (ARCANE_DEBUG est définie) ou
 * en mode \a check (ARCANE_CHECK est définie), met le programme en pause
 * pour éventuellement connecter un débugger dessus.
 *
 * En mode normal, lance une exception FatalErrorException avec le message
 * \a msg comme argument.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneDebugPause(const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double arg_value,const char* func_name);

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double arg_value1,long double arg_value2,const char* func_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signale un argument invalide d'une fonction mathématique.
 *
 * Une fois le message affiché, appelle arcaneDebugPause()
 *
 * \param arg_value valeur de l'argument invalide.
 * \param func_name nom de la fonction mathématique.
 */
ARCCORE_HOST_DEVICE inline void
arcaneMathError(long double arg_value,const char* func_name)
{
#ifndef ARCCORE_DEVICE_CODE
  _internalArcaneMathError(arg_value,func_name);
#else
  ARCANE_UNUSED(arg_value);
  ARCANE_UNUSED(func_name);
#endif
}

/*!
 * \brief Signale un argument invalide d'une fonction mathématique.
 *
 * Une fois le message affiché, appelle arcaneDebugPause()
 *
 * \param arg_value1 valeur du premier argument invalide.
 * \param arg_value2 valeur du second argument invalide.
 * \param func_name nom de la fonction mathématique.
 */
ARCCORE_HOST_DEVICE inline void
arcaneMathError(long double arg_value1,long double arg_value2,const char* func_name)
{
#ifndef ARCCORE_DEVICE_CODE
  _internalArcaneMathError(arg_value1,arg_value2,func_name);
#else
  ARCANE_UNUSED(arg_value1);
  ARCANE_UNUSED(arg_value2);
  ARCANE_UNUSED(func_name);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signale une fonction non implémentée.
 *
 * Une fois le message affiché, appelle arcaneDebugPause()
 *
 * \param file nom du fichier contenant la fonction
 * \param func nom de la fonction
 * \param numéro de ligne
 * \param msg message éventuel à afficher (0 si aucun)
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNotYetImplemented(const char* file,const char* func,unsigned long line,const char* msg);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Signale l'utilisation d'une fonction obsolète
extern "C++" ARCANE_UTILS_EXPORT void
arcaneDeprecated(const char* file,const char* func,unsigned long line,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utilisation d'un objet non référencé.
 *
 * Signale une tentative d'utilisation d'un objet qui ne devrait plus être
 * référencé. Affiche un message et appelle arcaneDebugPause() si demandé et
 * ensuite lance une exception FatalErrorException.
 *
 * \param ptr adresse de l'objet
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceError(const void* ptr);

/*!
 * \brief Utilisation d'un objet non référencé.
 *
 * Signale une tentative d'utilisation d'un objet qui ne devrait plus être
 * référencé. Affiche un message et appelle arcaneDebugPause() si demandé et
 * ensuite appelle std::terminate().
 *
 * \param ptr adresse de l'objet
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceErrorCallTerminate(const void* ptr);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned int size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(int size);

/*!
 * \brief Vérifie que \a ptr est aligné sur \a alignment octets.
 * Si ce n'est pas le cas, Sinon, lance une exception de type BadAlignmentException.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneCheckAlignment(const void* ptr,Integer alignment);

/*!
 * \brief Vrai si on est en mode vérification.
 *
 * Ce mode est actif si la macro ARCANE_CHECK est définie
 * ou si la méthode arcaneSetCheck() a été positionnée a vrai.
 */
extern "C++" ARCANE_UTILS_EXPORT 
bool arcaneIsCheck();

/*!
 * \brief Active ou désactive le mode vérification.
 *
 * Le mode vérification est toujours actif si la macro ARCANE_CHECK est définie.
 * Sinon, il est possible de l'activer avec cette méthode. Cela permet
 * d'activer certains tests même en mode optimisé.
 */
extern "C++" ARCANE_UTILS_EXPORT 
void arcaneSetCheck(bool v);

/*!
 * \brief Vrai si la macro ARCANE_DEBUG est définie
 */
extern "C++" ARCANE_UTILS_EXPORT
bool arcaneIsDebug();

/*!
 * \brief Vrai si arcane est compilé avec le support des threads ET qu'ils sont actifs
 */
extern "C++" ARCANE_UTILS_EXPORT 
bool arcaneHasThread();

/*!
 * \brief Active ou désactive le support des threads.
 *
 * Cette fonction ne doit être appelée que lors de l'initialisation
 * de l'application (ou avant) et ne pas être modifiée par la suite.
 * L'activation des threads n'est possible que si une implémentation
 * des threads existe sur la plate-forme et que Arcane a été compilé
 * avec ce support.
 */
extern "C++" ARCANE_UTILS_EXPORT 
void arcaneSetHasThread(bool v);

/*!
 * \brief Retourne l'identifiant du thread courant.
 *
 * Retourne toujours 0 si arcaneHasThread() est faux.
 */
extern "C++" ARCANE_UTILS_EXPORT
Int64 arcaneCurrentThread();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_DEBUG
extern "C++" ARCANE_UTILS_EXPORT bool _checkDebug(size_t);
#define ARCANE_DEBUGP(a,b)     if (_checkDebug(a)) { arcanePrintf b; }
#else
#define ARCANE_DEBUGP(a,b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#  define ARCANE_NOT_YET_IMPLEMENTED(a) \
{ arcaneNotYetImplemented(__FILE__,__PRETTY_FUNCTION__,__LINE__,(a)); }
#else
#  define ARCANE_NOT_YET_IMPLEMENTED(a) \
{ arcaneNotYetImplemented(__FILE__,"(NoInfo)",__LINE__,(a)); }
#endif

#define ARCANE_DEPRECATED ARCCORE_DEPRECATED

#define ARCANE_DEPRECATED_112 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_114 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_116 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_118 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_120 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_122 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_200 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_220 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_240 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_260 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_280 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_2018 ARCANE_DEPRECATED
#define ARCANE_DEPRECATED_2018_R(reason) [[deprecated(reason)]]

#ifndef ARCCORE_DEPRECATED_2021
#define ARCCORE_DEPRECATED_2021(reason) [[deprecated(reason)]]
#endif

#define ARCANE_DEPRECATED_REASON(reason) [[deprecated(reason)]]

#ifdef ARCANE_NO_DEPRECATED_LONG_TERM
#define ARCANE_DEPRECATED_LONG_TERM(reason)
#else
/*!
 * \brief Macro pour l'attribut 'deprecated' à long terme.
 *
 * Cette macro est pour indiquer les types ou fonctions
 * obsolète et donc qu'il est préférable de ne pas utiliser mais qui
 * ne seront pas supprimés avant plusieurs versions.
 */
#define ARCANE_DEPRECATED_LONG_TERM(reason) [[deprecated(reason)]]
#endif

// Définir cette macro si on souhaite supprimer de la compilation les
// méthodes et types obsolètes.
#define ARCANE_NO_DEPRECATED

// Si la macro est définie, ne notifie pas des méthodes obsolètes des anciennes
// classes tableaux.
#ifdef ARCANE_NO_NOTIFY_DEPRECATED_ARRAY
#define ARCANE_DEPRECATED_ARRAY
#else
#define ARCANE_DEPRECATED_ARRAY ARCANE_DEPRECATED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Les macros suivantes permettent de de créer un identifiant en suffixant
// le numéro de ligne du fichier. Cela permet d'avoir un identifiant unique
// pour un fichier et est utilisé par exemple pour générer des noms
// de variable globale pour l'enregistrement des services.
// La macro a utiliser est ARCANE_JOIN_WITH_LINE(name).
#define ARCANE_JOIN_HELPER2(a,b) a ## b
#define ARCANE_JOIN_HELPER(a,b) ARCANE_JOIN_HELPER2(a,b)
#define ARCANE_JOIN_WITH_LINE(a) ARCANE_JOIN_HELPER(a,__LINE__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// La macro ARCANE_NORETURN utilise l'attribut [[noreturn]] du C++11 pour
// indiquer qu'une fonction ne retourne pas.
#define ARCANE_NORETURN ARCCORE_NORETURN

//! Macro permettant de spécifier le mot-clé 'constexpr' du C++11
#define ARCANE_CONSTEXPR ARCCORE_CONSTEXPR

// Le C++11 définit un mot clé 'noexcept' pour indiquer qu'une méthode ne
// renvoie pas d'exceptions. Malheureusement, comme le support du C++11
// est fait de manière partielle par les compilateurs, cela ne marche pas
// pour tous. En particulier, icc 13, 14 et 15 ne supportent pas cela, ni
// Visual Studio 2013 et antérieurs.
#define ARCANE_NOEXCEPT ARCCORE_NOEXCEPT
#define ARCANE_NOEXCEPT_FALSE ARCCORE_NOEXCEPT_FALSE

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
#  define ARCANE_ALIGNAS(value) __declspec(align(value))
//! Macro pour garantir l'alignement d'une classe sur \a value octets
#  define ARCANE_ALIGNAS_PACKED(value) __declspec(align(value))
#else
//! Macro pour garantir le compactage et l'alignement d'une classe sur \a value octets
#  define ARCANE_ALIGNAS_PACKED(value) __attribute__ ((aligned (value),packed))
//! Macro pour garantir l'alignement d'une classe sur \a value octets
#  define ARCANE_ALIGNAS(value) __attribute__ ((aligned (value)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_SWIG
#ifdef ARCANE_DEPRECATED
#undef ARCANE_DEPRECATED
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_CHECK) || defined(ARCANE_DEBUG)
#ifndef ARCANE_DEBUG_ASSERT
#define ARCANE_DEBUG_ASSERT
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Signalue l'utilisation d'un pointeur nul.
 *
 * Signale une tentative d'utilisation d'un pointeur nul.
 * Affiche un message, appelle arcaneDebugPause() et lance une exception
 * de type FatalErrorException.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNullPointerError [[noreturn]] ();

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
 * via la macro ARCANE_CHECK_POINTER.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneThrowNullPointerError [[noreturn]] (const char* ptr_name,const char* text);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 */
static inline void
arcaneCheckNull(const void* ptr)
{
  if (!ptr)
    arcaneNullPointerError();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la taille avec padding pour une taille \a size.
 *
 * La valeurs retournée est un multiple de SIMD_PADDING_SIZE et vaut:
 * - 0 si \a size est inférieur ou égal à 0.
 * - \a size si \a size est un multiple de SIMD_PADDING_SIZE.
 * - le multiple de SIMD_PADDING_SIZE immédiatement supérieur à \a size sinon.
 */
extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneSizeWithPadding(Integer size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Macros utilisées pour le débug.
 */
#ifdef ARCANE_DEBUG_ASSERT
extern "C++" ARCANE_UTILS_EXPORT void _doAssert(const char*,const char*,const char*,size_t);
template<typename T> inline T*
_checkPointer(T* t,const char* file,const char* func,size_t line)
{
  if (!t){
    _doAssert("ARCANE_ASSERT",file,func,line);
    arcanePrintf("Bad Pointer");
  }
  return t;
}
#  ifdef __GNUG__
#    define ARCANE_D_WHERE(a)  Arcane::_doAssert(a,__FILE__,__PRETTY_FUNCTION__,__LINE__)
#    define ARCANE_DCHECK_POINTER(a) Arcane::_checkPointer((a),__FILE__,__PRETTY_FUNCTION__,__LINE__);
#  else
#    define ARCANE_D_WHERE(a)  Arcane::_doAssert(a,__FILE__,"(NoInfo)",__LINE__)
#    define ARCANE_DCHECK_POINTER(a) Arcane::_checkPointer((a),__FILE__,"(NoInfo"),__LINE__);
#  endif
#  define ARCANE_CHECK_PTR(a) \
   {if (!(a)){Arcane::arcanePrintf("Null value");ARCANE_D_WHERE("ARCANE_ASSERT");}}

#  define ARCANE_ASSERT(a,b) \
  {if (!(a)){ Arcane::arcanePrintf("Assertion '%s' fails:",#a); Arcane::arcanePrintf b; ARCANE_D_WHERE("ARCANE_ASSERT");}}
#  define ARCANE_WARNING(a) \
   { Arcane::arcanePrintf a; ARCANE_D_WHERE("ARCANE_WARNING"); }
#else
#  define ARCANE_CHECK_PTR(a)
#  define ARCANE_ASSERT(a,b)
#  define ARCANE_WARNING(a)
#  define ARCANE_DCHECK_POINTER(a) (a);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour envoyer une exception avec formattage.
 *
 * \a exception_class est le type de l'exception. Les arguments suivants de
 * la macro sont utilisés pour formatter un message d'erreur via la
 * méthode String::format().
 */
#define ARCANE_THROW(exception_class,...) \
  ARCCORE_THROW(exception_class,__VA_ARGS__)

/*!
 * \brief Macro pour envoyer une exception avec formattage si \a cond est vrai.
 *
 * \a exception_class est le type de l'exception. Les arguments suivants de
 * la macro sont utilisés pour formatter un message d'erreur via la
 * méthode String::format().
 *
 * \sa ARCANE_THROW
 */
#define ARCANE_THROW_IF(const, exception_class, ...)    \
  ARCCORE_THROW_IF(const, exception_class, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro envoyant une exception FatalErrorException.
 *
 * Les arguments de la macro sont utilisés pour formatter un message
 * d'erreur via la méthode String::format().
 */
#define ARCANE_FATAL(...) \
  ARCCORE_FATAL(__VA_ARGS__)

/*!
 * \brief Macro envoyant une exception FatalErrorException si \a cond est vrai
 *
 * Les arguments de la macro sont utilisés pour formatter un message
 * d'erreur via la méthode String::format().
 *
 * \sa ARCANE_FATAL
 */
#define ARCANE_FATAL_IF(const, ...) \
  ARCCORE_FATAL_IF(const, __VA_ARGS__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 *
 * Si le pointeur est nul, appelle arcaneThrowNullPointerError().
 * Sinon, retourne le pointeur.
 */
static inline void*
arcaneThrowIfNull(void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 *
 * Si le pointeur est nul, appelle à arcaneThrowNullPointerError().
 * Sinon, retourne le pointeur.
 */
static inline const void*
arcaneThrowIfNull(const void* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'un pointeur n'est pas nul.
 *
 * Si le pointeur est nul, appelle à arcaneThrowNullPointerError().
 * Sinon, retourne le pointeur.
 */
template<typename T> inline T*
arcaneThrowIfNull(T* ptr,const char* ptr_name,const char* text)
{
  if (!ptr)
    arcaneThrowNullPointerError(ptr_name,text);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro retournant le pointeur \a ptr s'il est non nul
 * ou lancant une exception s'il est nul.
 *
 * \sa arcaneThrowIfNull().
 */
#define ARCANE_CHECK_POINTER(ptr) \
  arcaneThrowIfNull(ptr,#ptr,nullptr)

/*!
 * \brief Macro retournant le pointeur \a ptr s'il est non nul
 * ou lancant une exception s'il est nul.
 *
 * \sa arcaneThrowIfNull().
 */
#define ARCANE_CHECK_POINTER2(ptr,text)\
  arcaneThrowIfNull(ptr,#ptr,text)

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
extern "C++" ARCANE_UTILS_EXPORT void
arcaneRangeError [[noreturn]] (Int64 i,Int64 max_size);

/*!
 * \brief Vérifie un éventuel débordement de tableau.
 */
static inline constexpr ARCCORE_HOST_DEVICE void
arcaneCheckAt(Int64 i,Int64 max_size)
{
#ifndef ARCCORE_DEVICE_CODE
  if (i<0 || i>=max_size)
    arcaneRangeError(i,max_size);
#else
  ARCANE_UNUSED(i);
  ARCANE_UNUSED(max_size);
#endif
}

#if defined(ARCANE_CHECK) || defined(ARCANE_DEBUG)
#define ARCANE_CHECK_AT(a,b) ::Arcane::arcaneCheckAt((a),(b))
#else
#define ARCANE_CHECK_AT(a,b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
