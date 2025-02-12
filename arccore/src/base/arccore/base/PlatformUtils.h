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
#ifndef ARCCORE_BASE_PLATFORMUTILS_H
#define ARCCORE_BASE_PLATFORMUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
class IStackTraceService;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Espace de nom pour les fonctions dépendant de la plateforme.
 * 
 * Cet espace de nom contient toutes les fonctions dépendant de la plateforme.
 */
namespace Arccore::Platform
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisations spécifiques à une platforme.
 *
 * Cette routine est appelé lors de l'initialisation de l'architecture.
 * Elle permet d'effectuer certains traitements qui dépendent de la
 * plateforme.
 *
 * Active les exceptions flottantes i elles sont disponibles.
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformInitialize();

/*!
 * \brief Initialisations spécifiques à une platforme.
 *
 * Cette routine est appelé lors de l'initialisation de l'architecture.
 * Elle permet d'effectuer certains traitements qui dépendent de la
 * plateforme.
 *
 * Si \a enable_fpe est vrai, les exceptions flottantes sont activées si elles
 * sont disponibles (via l'appel à enableFloatingException().
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformInitialize(bool enable_fpe);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Routines de fin de programme spécifiques à une platforme.
 *
 Cette routine est appelé juste avant de quitter le programme.
 */
extern "C++" ARCCORE_BASE_EXPORT void
platformTerminate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Date courante.
 *
 * La chaîne est retournée sous la forme jour/mois/année.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDate();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Date courante.
 *
 * Retourne la date courante, exprimée en secondes écoulées
 * depuis le 1er janvier 1970.
 */
extern "C++" ARCCORE_BASE_EXPORT long
getCurrentTime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Date et l'heure courante sous la forme ISO 8601.
 *
 * La chaîne est retournée sous la forme AAAA-MM-JJTHH:MM:SS.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDateTime();

/*!
 * \brief Nom de la machine sur lequel tourne le processus.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getHostName();

/*!
 * \brief Chemin du répertoire courant.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCurrentDirectory();

/*!
 * \brief Numéro du processus.
 */
extern "C++" ARCCORE_BASE_EXPORT int
getProcessId();

/*!
 * \brief Nom de l'utilisateur.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getUserName();

/*!
 * \brief Répertoire contenant les documents utilisateurs.
 *
 * Cela correspond à la variable d'environnement HOME sur Unix,
 * ou le répertoire 'Mes documents' sous Win32.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getHomeDirectory();

/*!
 * \brief Longueur du fichier \a filename.
 * Si le fichier n'est pas lisible ou n'existe pas, retourne 0.
 */
extern "C++" ARCCORE_BASE_EXPORT long unsigned int
getFileLength(const String& filename);

/*!
 * \brief Variable d'environnement du nom \a name.
 *
 * Si aucune variable de nom \a name n'est définie,
 * la chaîne nulle est retournée.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getEnvironmentVariable(const String& name);

/*!
 * \brief Créé un répertoire.
 *
 * Créé le répertoire de nom \a dir_name. Si nécessaire, créé les
 * répertoires parents nécessaires.
 *
 * \retval true en cas d'échec,
 * \retval false en cas de succès ou si le répertoire existe déjà.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
recursiveCreateDirectory(const String& dir_name);

/*!
 * \brief Créé le répertoire.
 *
 * Créé un répertoire de nom \a dir_name. Cette fonction suppose
 * que le répertoire parent existe déjà.
 *
 * \retval true en cas d'échec,
 * \retval false en cas de succès ou si le répertoire existe déjà.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
createDirectory(const String& dir_name);

/*!
 * \brief Supprime le fichier \a file_name.
 *
 * \retval true en cas d'échec,
 * \retval false en cas de succès ou si le fichier n'existe pas.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
removeFile(const String& file_name);

/*!
 * \brief Vérifie que le fichier \a file_name est accessible et lisible.
 *
 * \retval true si le fichier est lisible,
  * \retval false sinon.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
isFileReadable(const String& file_name);

/*!
 * \brief Retourne le nom du répertoire d'un fichier.
 *
 * Retourne le nom du répertoire dans lequel se trouve le fichier
 * de nom \a file_name.
 * Par exemple, si \a file_name vaut \c "/tmp/toto.cc", retourne "/tmp".
 * Si le fichier ne contient pas de répertoires, retourne \c ".".
 */
extern "C++" ARCCORE_BASE_EXPORT String
getFileDirName(const String& file_name);

/*!
 * \brief Copie de zone mémoire
 *
 * Copie \a len octets de l'adresse \a from à l'adresse \a to.
 */
extern "C++" ARCCORE_BASE_EXPORT void
stdMemcpy(void* to,const void* from,::size_t len);

/*!
 * \brief Mémoire utilisée em octets
 *
 * \return la mémoire utilisée ou un nombre négatif si inconnu
 */
extern "C++" ARCCORE_BASE_EXPORT double
getMemoryUsed();

/*!
 * \brief Temps CPU utilisé en microsecondes.
 *
 * L'origine du temps CPU est pris lors de l'appel à platformInitialize().
 *
 * \return le temps CPU utilisé en microsecondes.
 */
extern "C++" ARCCORE_BASE_EXPORT Int64
getCPUTime();

/*!
 * \brief Temps Real utilisé en secondes.
 *
 * \return le temps utilisé en seconde.
 */
extern "C++" ARCCORE_BASE_EXPORT Real
getRealTime();

/*!
 * \brief Retourne un temps sous forme des heures, minutes et secondes.
 *
 * Converti \a t, exprimé en seconde, sous la forme AhBmCs
 * avec A les heures, B les minutes et C les secondes.
 * Par exemple, 3732 devient 1h2m12s.
 */
extern "C++" ARCCORE_BASE_EXPORT String
timeToHourMinuteSecond(Real t);

/*!
 * \brief Retourne \a true si \a v est dénormalisé (flottant invalide).
 *  
 * Si la plate-forme ne supporte pas cette notion, retourne toujours \a false.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
isDenormalized(Real v);

/*!
 * \brief Service utilisé pour obtenir la pile d'appel.
 *
 * Peut retourner nul si aucun service n'est disponible.
 */
extern "C++" ARCCORE_BASE_EXPORT IStackTraceService*
getStackTraceService();

/*! \brief Positionne le service utilisé pour obtenir la pile d'appel.
  
  Retourne l'ancien service utilisé.
*/
extern "C++" ARCCORE_BASE_EXPORT IStackTraceService*
setStackTraceService(IStackTraceService* service);

/*!
 * \brief Retourne une chaîne de caractere contenant la pile d'appel.
 *
 * Si aucun service de gestion de pile d'appel n'est présent
 * (getStackTraceService()==0), la chaîne retournée est nulle.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getStackTrace();

/*
 * \brief Copie une chaîne de caractère avec vérification de débordement.
 *
 * \param input chaîne à copier.
 * \param output pointeur où sera recopié la chaîne.
 * \param output_len mémoire allouée pour \a output.
 */
extern "C++" ARCCORE_BASE_EXPORT void
safeStringCopy(char* output,Integer output_len,const char* input);

/*!
 * \brief Met le process en sommeil pendant \a nb_second secondes.
 */
extern "C++" ARCCORE_BASE_EXPORT void
sleep(Integer nb_second);

/*!
 * \brief Active ou désactive les exceptions lors d'un calcul flottant.
 * Cette opération n'est pas supportée sur toutes les plateformes. Dans
 * le cas où elle n'est pas supportée, rien ne se passe.
 */
extern "C++" ARCCORE_BASE_EXPORT void
enableFloatingException(bool active);

//! Indique si les exceptions flottantes du processeur sont activées.
extern "C++" ARCCORE_BASE_EXPORT bool
isFloatingExceptionEnabled();

/*!
 * \brief Lève une exception flottante.
 *
 * Cette méthode ne fait rien si hasFloatingExceptionSupport()==false.
 * En général sous Linux, cela se traduit par l'envoie d'un signal
 * de type SIGFPE. Par défaut %Arccore récupère ce signal et
 * lève une exception de type 'ArithmeticException'.
 */
extern "C++" ARCCORE_BASE_EXPORT void
raiseFloatingException();

/*!
 * \brief Indique si l'implémentation permet de modifier
 * l'état d'activation des exceptions flottantes.
 *
 * Si cette méthode retourne \a false, alors les méthodes
 * enableFloatingException() et isFloatingExceptionEnabled()
 * sont sans effet.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
hasFloatingExceptionSupport();

/*!
 * \brief Affiche la pile d'appel sur le flot \a ostr.
 */
extern "C++" ARCCORE_BASE_EXPORT void
dumpStackTrace(std::ostream& ostr);

/*!
 * \brief Indique si la console supporte les couleurs.
 */
extern "C++" ARCCORE_BASE_EXPORT bool
getConsoleHasColor();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Chaîne de caractère permettant d'identifier le compilateur
 * utilisé pour compiler %Arccore.
 */
extern "C++" ARCCORE_BASE_EXPORT String
getCompilerId();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace Platform = Arccore::Platform;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::platform
{

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

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

