﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PlatformUtils.h                                             (C) 2000-2019 */
/*                                                                           */
/* Fonctions utilitaires dépendant de la plateforme.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PLATFORMUTILS_H
#define ARCANE_UTILS_PLATFORMUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/PlatformUtils.h"
#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOnlineDebuggerService;
class IProfilingService;
class IProcessorAffinityService;
class IDynamicLibraryLoader;
class ISymbolizerService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Espace de nom pour les fonctions dépendant de la plateforme.
 
  Cet espace de nom contient toutes les fonctions dépendant de la plateforme.
*/
namespace platform
{

/*!
 * \brief Initialisations spécifiques à une platforme.
 *
 Cette routine est appelé lors de l'initialisation de l'architecture.
 Elle permet d'effectuer certains traitements qui dépendent de la
 plateforme
 */
extern "C++" ARCANE_UTILS_EXPORT void platformInitialize();

/*!
 * \brief Routines de fin de programme spécifiques à une platforme.
 *
 Cette routine est appelé juste avant de quitter le programme.
 */
extern "C++" ARCANE_UTILS_EXPORT void platformTerminate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arccore::Platform::getCurrentDate;
using Arccore::Platform::getCurrentTime;
using Arccore::Platform::getCurrentDateTime;
using Arccore::Platform::getHostName;
using Arccore::Platform::getCurrentDirectory;
using Arccore::Platform::getProcessId;
using Arccore::Platform::getUserName;
using Arccore::Platform::getHomeDirectory;
using Arccore::Platform::getFileLength;
using Arccore::Platform::getEnvironmentVariable;
using Arccore::Platform::recursiveCreateDirectory;
using Arccore::Platform::createDirectory;
using Arccore::Platform::removeFile;
using Arccore::Platform::isFileReadable;
using Arccore::Platform::getFileDirName;
using Arccore::Platform::stdMemcpy;
using Arccore::Platform::getMemoryUsed;
using Arccore::Platform::getCPUTime;
using Arccore::Platform::getRealTime;
using Arccore::Platform::timeToHourMinuteSecond;
using Arccore::Platform::isDenormalized;
using Arccore::Platform::safeStringCopy;
using Arccore::Platform::sleep;

using Arccore::Platform::enableFloatingException;
using Arccore::Platform::isFloatingExceptionEnabled;
using Arccore::Platform::raiseFloatingException;
using Arccore::Platform::hasFloatingExceptionSupport;

using Arccore::Platform::getStackTraceService;
using Arccore::Platform::setStackTraceService;
using Arccore::Platform::getStackTrace;
using Arccore::Platform::dumpStackTrace;

using Arccore::Platform::getConsoleHasColor;
using Arccore::Platform::getCompilerId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service utilisé pour obtenir des informations
 * sur les symboles du code source.
 *
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT ISymbolizerService*
getSymbolizerService();

/*!
 * \brief Positionne le service pour obtenir des informations
 * sur les symboles du code source.
 *
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT ISymbolizerService*
setSymbolizerService(ISymbolizerService* service);

/*!
 * \brief Service utilisé pour la gestion de l'affinité des processeurs.
 *
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IProcessorAffinityService*
getProcessorAffinityService();

/*!
 * \brief Positionne le service utilisé pour la gestion de l'affinité des processeurs.
 *
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IProcessorAffinityService*
setProcessorAffinityService(IProcessorAffinityService* service);

/*!
 * \brief Service utilisé pour obtenir pour obtenir des informations de profiling.
 * 
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IProfilingService*
getProfilingService();

/*!
 * \brief Positionne le service utilisé pour obtenir des informations de profiling.
 * 
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IProfilingService*
setProfilingService(IProfilingService* service);

/*!
 * \brief Service utilisé pour obtenir la mise en place d'une architecture en ligne de debug.
 * 
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IOnlineDebuggerService*
getOnlineDebuggerService();

/*!
 * \brief Positionne le service a utiliser pour l'architecture en ligne de debug.
 * 
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IOnlineDebuggerService*
setOnlineDebuggerService(IOnlineDebuggerService* service);

/*!
 * \brief Service utilisé pour gérer les threads.
 * 
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IThreadImplementation*
getThreadImplementationService();

/*!
 * \brief Positionne le service utilisé pour gérer les threads.
 * 
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IThreadImplementation*
setThreadImplementationService(IThreadImplementation* service);

/*!
 * \brief Service utilisé pour charger dynamiquement des bibliothèques.
 * 
 * Peut retourner \c nullptr si le chargement dynamique n'est pas disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IDynamicLibraryLoader*
getDynamicLibraryLoader();

/*!
 * \brief Positionne le service utilisé pour charger dynamiquement des bibliothèques.
 * 
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IDynamicLibraryLoader*
setDynamicLibraryLoader(IDynamicLibraryLoader* idll);

/*!
 * \brief Remet à timer d'alarme à \a nb_second.
 *
 * Le timer déclenchera un signal (SIGALRM) au bout de \a nb_second.
 */
extern "C++" ARCANE_UTILS_EXPORT void
resetAlarmTimer(Integer nb_second);

/*!
 * \brief Vrai si le code s'exécute avec le runtime .NET.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
hasDotNETRuntime();

/*!
 * \brief Positionne si le code s'exécute avec le runtime .NET.
 *
 * Cette fonction ne peut être positionnée qu'au démarrage
 * du calcul avant arcaneInitialize().
 */
extern "C++" ARCANE_UTILS_EXPORT void
setHasDotNETRuntime(bool v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Appelle le Garbage Collector de '.Net' s'il est disponible
extern "C++" ARCANE_UTILS_EXPORT void
callDotNETGarbageCollector();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur spécifique pour les accélérateurs.
 *
 * Si non nul, cet allocateur permet d'allouer de la mémoire sur l'hôte en
 * utilisant le runtime spécique de l'allocateur.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getAcceleratorHostMemoryAllocator();

/*!
 * \brief Positionne l'allocateur spécifique pour les accélérateurs.
 *
 * Retourne l'ancien allocateur utilisé. L'allocateur spécifié doit rester
 * valide durant toute la durée de vie de l'application.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
setAcceleratorHostMemoryAllocator(IMemoryAllocator* a);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur par défaut pour les données.
 *
 * Cette allocateur utilise celui getAcceleratorHostMemoryAllocator()
 * s'il est disponible, sinon il utilise un allocateur aligné.
 *
 * Il est garanti que l'allocateur retourné permettra d'utiliser la donnée
 * sur accélerateur si cela est disponible.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator*
getDefaultDataAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le gestionnaire de ressource mémoire pour les données.
 *
 * Le gestionnaire doit rester valide durant toute l'exécution du programme.
 *
 * Retourne l'ancien gestionnaire.
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng*
setDataMemoryRessourceMng(IMemoryRessourceMng* mng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de ressource mémoire pour les données.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng*
getDataMemoryRessourceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit le contenu d'un fichier et le conserve dans \a out_bytes.
 *
 * Lit le fichier de nom \a filename et remplit \a out_bytes avec le contenu
 * de ce fichier. Si \a is_binary est vrai, le fichier est ouvert en mode
 * binaire. Sinon il est ouvert en mode texte.
 *
 * \retval true en cas d'erreur
 * \retval false sinon.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
readAllFile(StringView filename, bool is_binary, ByteArray& out_bytes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit le contenu d'un fichier et le conserve dans \a out_bytes.
 *
 * Lit le fichier de nom \a filename et remplit \a out_bytes avec le contenu
 * de ce fichier. Si \a is_binary est vrai, le fichier est ouvert en mode
 * binaire. Sinon il est ouvert en mode texte.
 *
 * \retval true en cas d'erreur
 * \retval false sinon.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
readAllFile(StringView filename, bool is_binary, Array<std::byte>& out_bytes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le nom complet avec le chemin de l'exécutable.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getExeFullPath();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le chemin complet d'une bibliothèque dynamique chargée.
 *
 * Retourne le chemin complet de la bibliothèque dynamique de nom
 * \a dll_name. \a dll_name doit contenir juste le nom de la bibliothèque
 * sans les extensions spécifiques à la plateforme. Par exemple, sous Linux,
 * il ne faut pas mettre 'libtoto.so' mais juste 'toto'.
 *
 * Retourne une chaîne nulle si le chemin complet ne peut
 * par être déterminé.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getLoadedSharedLibraryFullPath(const String& dll_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Définition du pragma pour indiquer l'indépendance des itérations

/*!
 * \def ARCANE_PRAGMA_IVDEP
 * Pragma pour indiquer au compilateur que les itérations d'une boucle sont
 * indépendanntes. Ce pragma se positionne avant une boucle 'for'.
 */

/*!
 * \def ARCANE_PRAGMA_IVDEP_VALUE
 * Valeur du pragma ARCANE_PRAGMA_IVDEP
 */

// Pour les définitions, il faut finir par GCC car Clang et ICC définissent
// la macro __GNU__
// Pour CLANG, il n'y a pas encore d'équivalent au pragma ivdep de ICC.
// Celui qui s'en approche le plus est:
//   #pragma clang loop vectorize(enable)
// mais il ne force pas la vectorisation.
#ifdef __clang__
#  define ARCANE_PRAGMA_IVDEP_VALUE "clang loop vectorize(enable)"
#else
#  ifdef __INTEL_COMPILER
#    define ARCANE_PRAGMA_IVDEP_VALUE "ivdep"
#  else
#    ifdef __GNUC__
#      if (__GNUC__>=5)
#        define ARCANE_PRAGMA_IVDEP_VALUE "GCC ivdep"
#      endif
#    endif
#  endif
#endif

#ifdef ARCANE_PRAGMA_IVDEP_VALUE
#define ARCANE_PRAGMA_IVDEP _Pragma(ARCANE_PRAGMA_IVDEP_VALUE)
#else
#define ARCANE_PRAGMA_IVDEP
#define ARCANE_PRAGMA_IVDEP_VALUE ""
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace platform

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

