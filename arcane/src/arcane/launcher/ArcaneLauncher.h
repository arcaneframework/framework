// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneLauncher.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classe gérant l'exécution.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_ARCANELAUNCHER_H
#define ARCANE_LAUNCHER_ARCANELAUNCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

// Les fichiers suivants ne sont pas directement utilisés dans ce '.h'
// mais sont ajoutés pour que le code utilisateur n'ait besoin d'inclure
// que 'ArcaneLauncher.h'.
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/core/ApplicationBuildInfo.h"
#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arcane/core/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/launcher/DirectExecutionContext.h"
#include "arcane/launcher/DirectSubDomainExecutionContext.h"
#include "arcane/launcher/IDirectExecutionContext.h"
#include "arcane/launcher/StandaloneAcceleratorMng.h"
#include "arcane/launcher/StandaloneSubDomain.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMainFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de gestion de l'exécution.
 *
 * Il existe deux modes d'utilisation d'%Arcane : le mode classique et le mode
 * autonome.
 *
 * Quel que soit le mode retenu, la première chose à faire est d'initialiser %Arcane en
 * positionnant les arguments via la méthode init() car certains paramètres de la
 * ligne de commande sont utilisés pour remplir les propriétés
 * de applicationInfo() et dotNetRuntimeInitialisationInfo().
 *
 * La page \ref arcanedoc_execution_launcher donne des exemples d'usage.
 *
 * Les deux modes d'éxécutions sont:
 * - le mode classique qui utilise une boucle en temps et donc l'exécution
 *   complète sera gérée par %Arcane. Dans mode il suffit d'appeler
 *   la méthode run() sans arguments.
 * - le mode autonome qui permet d'utiliser %Arcane sous la forme d'une bibliothèque.
 *   Pour ce mode il faut utiliser la méthode createStandaloneSubDomain()
 *   ou createStandaloneAcceleratorMng(). La page \ref arcanedoc_execution_direct_execution
 *   décrit comment utiliser ce mécanisme.
 *
 * L'usage classique est le suivant:
 *
 * \code
 * int main(int* argc,char* argv[])
 * {
 *   ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
 *   auto& app_info = ArcaneLauncher::applicationInfo();
 *   app_info.setCodeName("MyCode");
 *   app_info.setCodeVersion(VersionInfo(1,0,0));
 *   return ArcaneLauncher::run();
 * }
 * \endcode
 *
 * L
 */
class ARCANE_LAUNCHER_EXPORT ArcaneLauncher
{
  friend StandaloneSubDomain;

 public:

  /*!
   * \brief Positionne les informations à partir des arguments de la ligne
   * de commande et initialise le lanceur.
   *
   * Cette méthode remplit les valeurs non initialisées
   * de applicationInfo() et dotNetRuntimeInitialisationInfo() avec
   * les paramètres spécifiés dans \a args.
   *
   * Il ne faut appeler cette méthode qu'une seule fois. Les appels supplémentaires
   * génèrent une exception FatalErrorException.
   */
  static void init(const CommandLineArguments& args);

  /*!
   * \brief Indique si init() a déjà été appelé.
   */
  static bool isInitialized();

  /*!
   * \brief Point d'entrée de l'exécutable dans Arcane.
   *
   * Cette méthode appelle initialise l'application, lit le jeu de données
   * et exécute le code suivant la boucle en temps spécifiée dans le jeu de donnée.
   *
   * \retval 0 en cas de succès
   * \return une valeur différente de 0 en cas d'erreur.
   */
  static int run();

 /*!
   * \brief Exécution directe.
   *
   * Initialise l'application et appelle la fonction \a func après l'initialisation
   * Cette méthode ne doit être appelée qu'en exécution séquentielle.
   */
  static int run(std::function<int(DirectExecutionContext&)> func);

 /*!
   * \brief Exécution directe avec création de sous-domaine.
   *
   * Initialise l'application et créé le ou les sous-domaines et appelle
   * la fonction \a func après.
   * Cette méthode permet d'exécuter du code sans passer par les mécanismes
   * de la boucle en temps.
   * Cette méthode permet de gérer automatiquement la création des sous-domaines
   * en fonction des paramètres de lancement (exécution parallèle MPI, multithreading, ...).
   */
  static int run(std::function<int(DirectSubDomainExecutionContext&)> func);

  /*!
   * \brief Positionne la fabrique par défaut pour créer les différents gestionnaires
   *
   * Cette méthode doit être appelée avant run(). L'instance passée en argument doit
   * rester valide durant l'exécution de run(). L'appelant reste propriétaire
   * de l'instance.
   */
  static void setDefaultMainFactory(IMainFactory* mf);

 /*!
   * \brief Informations sur l'application.
   *
   * Cette méthode permet de récupérer l'instance de `ApplicationInfo`
   * qui sera utilisée lors de l'appel à run().
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à runDirect().
   */
  static ApplicationInfo& applicationInfo();

 /*!
   * \brief Informations sur les paramêtre d'exécutions de l'application.
   *
   * Cette méthode permet de récupérer l'instance de `ApplicationBuildInfo`
   * qui sera utilisée lors de l'appel à run().
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à runDirect().
   */
  static ApplicationBuildInfo& applicationBuildInfo();

  /*!
   * \brief Informations pour l'initialisation du runtime '.Net'.
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à rundDirect().
   */
  static DotNetRuntimeInitialisationInfo& dotNetRuntimeInitialisationInfo();

  /*!
   * \brief Informations pour l'initialisation des accélerateurs.
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à rundDirect().
   */
  static AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo();

  //! Nom complet du répertoire où se trouve l'exécutable
  static String getExeDirectory();

  /*!
   * \brief Créé une implémentation autonome pour gérer les accélérateurs.
   *
   * Il faut appeler init() avant d'appeler cette méthode. Le choix du
   * runtime (Arcane::Accelerator::eExecutionPolicy) est déterminé
   * par les arguments utilisés lors de l'appel à init() ou spécifiés via
   * acceleratorRuntimeInitialisationInfo() (voir
   * \ref arcanedoc_parallel_accelerator_exec pour plus d'informations)
   */
  static StandaloneAcceleratorMng createStandaloneAcceleratorMng();

  /*!
   * \brief Créé une implémentation autonome pour gérer un sous-domaine.
   *
   * Une seule instance de StandaloneSubDomain est autorisée. Si on
   * appelle cette méthode plus d'une fois cela génère une exception.
   *
   * Il faut appeler init() avant d'appeler cette méthode.
   *
   * Si on appelle cette méthode il ne faut pas appeler d'autres méthodes
   * d'exécution de ArcaneLauncher (par exemple ArcaneLauncher::run()).
   *
   * \a case_file_name est le nom du fichier contenant le jeu de données
   * Si nul, alors il n'y a pas de jeu de données.
   */
  static StandaloneSubDomain createStandaloneSubDomain(const String& case_file_name);

  /*!
   * \brief Demande d'aide avec l'option "--help" ou "-h".
   *
   * Méthode permettant de savoir si l'utilisateur a demandé l'aide
   * avec l'option "--help" ou "-h".
   *
   * \return true si l'aide a été demandée.
   */
  static bool needHelp();

  /*!
   * \brief Affichage de l'aide générique Arcane.
   *
   * Méthode permettant d'afficher l'aide générique Arcane si
   * l'utilisateur l'a demandée avec l'option "--help" ou "-h".
   *
   * \return true si l'aide a été demandée.
   */
  static bool printHelp();

 public:

  /*!
   * \deprecated
   */
  ARCCORE_DEPRECATED_2020("Use run(func) instead")
  static int runDirect(std::function<int(IDirectExecutionContext*)> func);

  /*!
   * \deprecated
   */
  ARCCORE_DEPRECATED_2020("Use init(args) instead")
  static void setCommandLineArguments(const CommandLineArguments& args)
  {
    init(args);
  }

 private:

  static void _initStandalone();
  static void _notifyRemoveStandaloneSubDomain();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
