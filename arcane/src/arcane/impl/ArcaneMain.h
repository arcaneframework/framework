// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMain.h                                                (C) 2000-2023 */
/*                                                                           */
/* Classe gérant l'exécution.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ARCANEMAIN_H
#define ARCANE_IMPL_ARCANEMAIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/IArcaneMain.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfo;
class IMainFactory;
class IApplication;
class ICodeService;
class ServiceFactoryInfo;
class ArcaneMainExecInfo;
class DotNetRuntimeInitialisationInfo;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT ArcaneMainExecutionOverrideFunctor
{
  friend class ArcaneMain;
  friend class ArcaneMainExecInfo;
 public:
  explicit ArcaneMainExecutionOverrideFunctor(IFunctor* functor)
  : m_functor(functor), m_application(nullptr){}
  IFunctor* functor() { return m_functor; }
  IApplication* application() { return m_application; }
 private:
  IFunctor* m_functor;
  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT IApplicationBuildInfoVisitor
{
 public:
  virtual ~IApplicationBuildInfoVisitor(){}
 public:
  virtual void visit(ApplicationBuildInfo& app_build_info) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de gestion de l'exécution.
 *
 * Cette classe est interne à %Arcane et ne doit pas être utilisée
 * directement. Pour initialiser et exécuter le code il faut utiliser la
 * classe ArcaneLauncher.
 */
class ARCANE_IMPL_EXPORT ArcaneMain
: public IArcaneMain
{
  friend class ArcaneMainExecInfo;
  friend class ArcaneLauncher;
  friend class ArcaneMainAutoDetectRuntimeHelper;
  class Impl;

 public:
  // TODO: a supprimer.
  ArcaneMain(const ApplicationInfo& infos,IMainFactory* factory);
 public:
  ArcaneMain(const ApplicationInfo& app_info,IMainFactory* factory,
             const ApplicationBuildInfo& app_build_info,
             const DotNetRuntimeInitialisationInfo& dotnet_init_info,
             const AcceleratorRuntimeInitialisationInfo& accelerator_init_info);
  ~ArcaneMain() override;

 public:

  /*!
   * \brief Point d'entrée de l'exécutable dans Arcane.
   *
   * \note Cette méthode ne doit pas être appelée directement. Il est
   * préférable d'utiliser la classe ArcaneLauncher pour gérer le
   * lancement d'une exécution.
   *
   * Cette méthode effectue les appels suivants:
   *
   *  - création d'une instance <tt>a</tt> de IArcaneMain par l'appel
   *    à createArcaneMain().
   *  - contruit <tt>a</tt> par la méthode IArcaneMain::build() 
   *  - initialize <tt>a</tt> par la méthode IArcaneMain::initialize()
   *  - lance l'exécution par la méthode IArcaneMain::execute().
   *
   * \param app_info informations sur l'exécutable.
   * \param factory fabrique des gestionnaires de l'architecture. Si nul,
   * utilise la fabrique spécifiée par setDefaultMainFactory() sinon
   * une fabrique par défaut est utilisée.
   *
   * L'appel à cette méthode doit être précédé de Initialize();
   *
   * \retval 0 si l'exécution s'est déroulée sans erreur
   * \retval 1 en cas d'erreur inconnue.
   * \retval 2 en cas d'exception standard (std::exception)
   * \retval 3 en cas d'exception de l'architecture (IArcaneException)
   * \retval 4 en cas d'erreur fatale dans Arcane.
   *
   */
  static int arcaneMain(const ApplicationInfo& app_info,IMainFactory* factory=nullptr);

  /*!
   * \brief Point d'entrée de l'exécutable dans Arcane.
   *
   * Cette méthode appelle arcaneMain(const ApplicationInfo&,IMainFactory*)
   * en utilisant les valeurs de defaultApplicationInfo() et de la fabrique spécifiée
   * lors des appels à setDefaultMainFactory().
   */
  static int run();

   /*!
   * \brief Initialise Arcane.
   *
   * Cette méthode doit être appelée avant tout utilisation d'un objet de
   * Arcane. Elle peut être appelée plusieurs fois, auquel cas la méthode
   * arcaneFinalize() doit être appelée un nombre de fois équivalent.
   *
   * L'appel à run() provoque l'initialisation. Il n'est donc en général
   * pas nécessaire de faire appel à cette méthode.
   */
  static void arcaneInitialize();
  
  /*!
   * \brief Termine l'utilisation Arcane.
   *
   * Cette méthode doit être appelée à la fin de l'exécution. Une fois
   * appelée, les objets de Arcane ne doivent plus être utilisés.
   *
   * L'appel à run() gère l'initialisation et l'appel à cette méthode.
   * Il n'est donc en général pas nécessaire de faire appel directement
   * à cette méthode.
   *
   * \sa arcaneInitialize();
   */
  static void arcaneFinalize();

  /*!
   * \brief Indique que certains objets sont gérés par un ramasse-miette.
   *
   * Cette propriété ne peut être positionnée qu'au démarrage du calcul,
   * avant l'appel à arcaneInitialize().
   */
  static void setHasGarbageCollector();

  /*!
   * \brief Indique que l'on tourne dans le runtime .NET.
   *
   * Cette propriété ne peut être positionnée qu'au démarrage du calcul,
   * avant l'appel à arcaneInitialize().
   */
  static void setHasDotNETRuntime();

  /*!
   * \brief Positionne la fabrique par défaut.
   *
   * Cette méthode positionne la fabrique par défaut utilisée si aucune
   * n'est spécifiée dans l'appel à arcaneMain().
   *
   * Cette méthode doit être appelée avant arcaneMain().
   */
  static void setDefaultMainFactory(IMainFactory* mf);

  /*!
   * \brief Infos par défaut de l'application
   *
   * Cette méthode permet de récupérer l'instance de `ApplicationInfo`
   * qui sera utilisée lors de l'appel à arcaneMain() sans arguments.
   *
   * Il faut donc en général appeler cette méthode
   * avant l'appel à run().
   */
  static ApplicationInfo& defaultApplicationInfo();

  /*!
   * \brief Informations pour l'initialisation du runtime '.Net'.
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run().
   */
  static DotNetRuntimeInitialisationInfo& defaultDotNetRuntimeInitialisationInfo();

  /*!
   * \brief Informations pour l'initialisation des accélerateurs.
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à rundDirect().
   */
  static AcceleratorRuntimeInitialisationInfo& defaultAcceleratorRuntimeInitialisationInfo();

  /*!
   * \brief Informations pour l'initialisation des accélerateurs.
   *
   * Pour être prise en compte, ces informations doivent être modifiées
   * avant l'appel à run() ou à rundDirect().
   */
  static ApplicationBuildInfo& defaultApplicationBuildInfo();

  /*!
   * \brief Appelle le fonctor \a functor en récupérant les éventuelles
   * exceptions.
   *
   * En retour \a clean_abort est vrai si le code s'arrête proprement,
   * c'est à dire en parallèle que l'ensemble des processus et des threads
   * exécutent le même code. C'est le cas par exemple si tous les procs
   * détectent la même erreur et lancent par exemple un ParallelFatalErrorException.
   * Dans ce cas, \a is_print indique si ce processus ou ce thread affiche
   * les messages d'erreur. Si \a is_print est vrai, le message d'erreur est
   * affiché sinon il ne l'est pas.
   *
   * Si \ a clean_abort est faux, cela signifie que l'un des processus ou 
   * thread s'arrête sans que les autres ne le sachent, ce qui en générale
   * se termine par MPI_Abort en parallèle.
   */
  
  static int callFunctorWithCatchedException(IFunctor* functor,IArcaneMain* amain,
                                             bool* clean_abort,
                                             bool is_print=true);

  /*!
   * brief Fonctor d'exécution.
   *
   * Cette méthode optionnelle permet de positionner un fonctor qui sera appelée
   * à la place de execute(). Ce fonctor est appelé une fois l'application
   * initialisée. 
   *
   * Comme l'appel à ce fonctor remplace l'exécution normale,
   * seule une instance de IApplication est disponible.
   * Il n'y a ni sous-domaine, ni session, ni maillage de disponible.
   *
   */  
  static void setExecuteOverrideFunctor(ArcaneMainExecutionOverrideFunctor* functor);

  //! Indique si on exécute une assembly '.Net' depuis un `main` en C++.
  static bool hasDotNetWrapper();

  /*!
   * \brief Retourne le temps (en seconde) pour l'initialisation
   * des runtimes accélérateurs pour ce processus.
   *
   * Retourne 0.0 si aucun runtime accélérateur n'a pas été initialisé.
   */
  static Real initializationTimeForAccelerator();

 public:
  
  /*!
   * \brief Ajoute une fabrique de service.
   *
   * Cette méthode doit être appelée avant arcaneMain()
   */
  static void addServiceFactoryInfo(IServiceFactoryInfo* factory);

  /*!
   * \brief Ajoute une fabrique de module
   *
   * Cette méthode doit être appelée avant arcaneMain()
   */
  static void addModuleFactoryInfo(IModuleFactoryInfo* factory);

 public:

  /*!
   * \brief Ajoute un visiteur pour remplir ApplicationBuildInfo.
   *
   * Le pointeur passé en argument doit rester valide jusqu'à l'appel à arcaneMain();
   * Les visiteurs enregistrés sont appelés juste avant de créer l'application.
   */
  static void addApplicationBuildInfoVisitor(IApplicationBuildInfoVisitor* visitor);

 public:
  
  static void redirectSignals();
  static bool isMasterIO() { return m_is_master_io; }
  static void setUseTestLogger(bool v);

 public:

  void build() override;
  void initialize() override;
  bool parseArgs(StringList args) override;
  int execute() override;
  void doAbort() override;
  void setErrorCode(int errcode) override;
  int errorCode() const override { return m_error_code; }
  void finalize() override {}

 public:

  const ApplicationInfo& applicationInfo() const override;
  const ApplicationBuildInfo& applicationBuildInfo() const override;
  const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const override;
  const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const override;
  IMainFactory* mainFactory() const override { return m_main_factory; }
  IApplication* application() const override { return m_application; }
  ServiceFactoryInfoCollection registeredServiceFactoryInfos() override;
  ModuleFactoryInfoCollection registeredModuleFactoryInfos() override;
  bool hasGarbageCollector() const override { return m_has_garbage_collector; }
  void setDirectExecuteFunctor(IDirectSubDomainExecuteFunctor* f) override { m_direct_sub_domain_execute_functor = f; }
  IDirectSubDomainExecuteFunctor* _directExecuteFunctor() const { return m_direct_sub_domain_execute_functor; }

 protected:

  IApplication* _application() { return m_application; }
  ApplicationBuildInfo& _applicationBuildInfo();
  static int _internalRun(IDirectSubDomainExecuteFunctor* func);

 private:

  Impl* m_p;
  IMainFactory* m_main_factory = nullptr;
  IApplication* m_application = nullptr;
  int m_error_code = 0;
  IDirectSubDomainExecuteFunctor* m_direct_sub_domain_execute_functor = nullptr;
  static bool m_has_garbage_collector;
  static bool m_is_master_io;
  static bool m_is_use_test_logger;
  static IMainFactory* m_default_main_factory;
  static ArcaneMainExecutionOverrideFunctor* m_exec_override_functor;

 private:

  static int _arcaneMain(const ApplicationInfo&,IMainFactory*);
  void _dumpHelp();
  void _parseApplicationBuildInfoArgs();
  //! Nombre de fois que arcaneInitialize() a été appelé
  static std::atomic<Int32> m_nb_arcane_init;
  //! 1 si init terminé, 0 sinon
  static std::atomic<Int32> m_is_init_done;
  static void _launchMissingInitException();
  static void _checkHasInit();
  static int _runDotNet();
  static void _checkAutoDetectMPI();
  static int _checkAutoDetectAccelerator(bool& has_accelerator);
  static void _setArcaneLibraryPath();
  static int _initRuntimes();
  static int _checkTestLoggerResult();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
