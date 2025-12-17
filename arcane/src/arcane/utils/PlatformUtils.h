// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PlatformUtils.h                                             (C) 2000-2025 */
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
 * \brief Positionne le service utilisé pour gérer les compteurs interne du processeur.
 *
 * Retourne l'ancien service utilisé.
 */
extern "C++" ARCANE_UTILS_EXPORT IPerformanceCounterService*
setPerformanceCounterService(IPerformanceCounterService* service);

/*!
 * \brief Service utilisé pour obtenir pour obtenir les compteurs interne du processeur.
 *
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCANE_UTILS_EXPORT IPerformanceCounterService*
getPerformanceCounterService();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
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
 * \deprecated Use MemoryUtils::getDefaultDataAllocator() instead.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: Use MemoryUtils::getDefaultDataAllocator() instead.")
ARCANE_UTILS_EXPORT IMemoryAllocator*
getAcceleratorHostMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne l'allocateur spécifique pour les accélérateurs.
 *
 * Retourne l'ancien allocateur utilisé. L'allocateur spécifié doit rester
 * valide durant toute la durée de vie de l'application.
 *
 * \deprecated Cette méthode est interne à Arcane.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IMemoryAllocator*
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
 *
 * * \deprecated Cette méthode est interne à Arcane.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
ARCANE_UTILS_EXPORT IMemoryRessourceMng*
setDataMemoryRessourceMng(IMemoryRessourceMng* mng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de ressource mémoire pour les données.
 *
 * Il est garanti que l'alignement est au moins celui retourné par
 * AlignedMemoryAllocator::Simd().
 *
 * \deprecated Cette méthode est interne à Arcane.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. Use methods from MemoryUtils instead.")
ARCANE_UTILS_EXPORT IMemoryRessourceMng*
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
/*!
 * \brief Remplit \a arg_list avec les arguments de la ligne de commande.
 *
 * Cette fonction remplit \a arg_list avec les arguments utilisés dans
 * l'appel à main().
 *
 * Actuellement cette méthode ne fonctionne que sous Linux. Pour les autres
 * plateforme elle retourne une liste vide.
 */
extern "C++" ARCANE_UTILS_EXPORT void
fillCommandLineArguments(StringList& arg_list);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère la pile d'appel via gdb.
 *
 * Cette méthode ne fonctionne que sous Linux et si GDB est installé. Dans
 * les autres cas c'est la chaîne nulle qui est retournée.
 *
 * Cette méthode appelle la commande std::system() pour lancer gbd qui doit
 * se trouver dans le PATH. Comme gdb charge ensuite les symboles de debug
 * la commande peut être assez longue à s'exécuter.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getGDBStack();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère la pile d'appel via lldb.
 *
 * Cette méthode est similaire à getGDBStack() mais utilise 'lldb' pour
 * récupérer la pile d'appel. Si `dotnet-sos` est installé, cela permet
 * aussi de récupérer les informations sur les méthodes du runtime 'dotnet'.
 */
extern "C++" ARCANE_UTILS_EXPORT String
getLLDBStack();

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
