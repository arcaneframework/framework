// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IArcaneMain.h                                               (C) 2000-2021 */
/*                                                                           */
/* Interface de la classe ArcaneMain.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IARCANEMAIN_H
#define ARCANE_IARCANEMAIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfo;
class ApplicationBuildInfo;
class IMainFactory;
class DotNetRuntimeInitialisationInfo;
class IDirectSubDomainExecuteFunctor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la classe de gestion du code.
 *
 Cette classe virtuelle sert à la création et l'initialisation des instances
 des gestionnaires du code. Elle pilote aussi le déroulement d'un cas.
 
 Une instance de cette classe est créée par l'intermédiaire de la méthode
 IMainFactory::createArcaneMain(), appelée par
 IMainFactory::arcaneMain().

 * L'implémentation doit au moins prendre en compte les aspects suivants.
 * - Analyser la ligne de commande.
 * - Créer une instance d'un superviseur (IMainFactory::createSuperMng()),
 * la construire (ISuperMng::build()) et l'initialiser (ISuperMng::initialize()).
 * - Créer une instance du chargeur de module (IMainFactory::createModuleLoader())
 */
class ARCANE_CORE_EXPORT IArcaneMain
{
 public:

  //! Libère les ressources.
  virtual ~IArcaneMain() {}

 public:

  /*!
   * Récupère l'instance globale.
   *
   * \warning L'instance globale n'est disponible que pendant l'appel à
   * ArcaneMain::arcaneMain().
   */
  static IArcaneMain* arcaneMain();
  /*!
   * \internal.
   */
  static void setArcaneMain(IArcaneMain* arcane_main);

 private:

  static IArcaneMain* global_arcane_main;

 public:

  /*!
   * \brief Construit les membres la classe.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée. Cette méthode doit être appelée avant initialize().
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void build() =0;

  /*!
   * \brief Initialise l'instance.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée.
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void initialize() =0;

 public:

  /*! \brief Analyse les arguments.
   *
   * Les arguments reconnus doivent être supprimés de la liste.
   *
   * \retval true si l'exécution doit s'arrêter,
   * \retval false si elle continue normalement
   */
  virtual bool parseArgs(StringList args) =0;

  /*! \brief Lance l'exécution.
   * Cette méthode ne retourne que lorsqu'on quitte le programme.
   * \return le code de retour d'Arcane, 0 si tout est ok.
   */
  virtual int execute() =0;

  //! Effectue les dernières opérations avant destruction de l'instance
  virtual void finalize() =0;
  
  //! Code d'erreur de l'exécution
  virtual int errorCode() const =0;
  
  //! Positionne le code de retour
  virtual void setErrorCode(int errcode) =0;

  //! Effectue un abort.
  virtual void doAbort() =0;

 public:

  //! Informations sur l'éxécutable
  virtual const ApplicationInfo& applicationInfo() const =0;

  //! Informations pour construire l'instance IApplication.
  virtual const ApplicationBuildInfo& applicationBuildInfo() const =0;

  //! Informations d'initialisation du runtime '.Net'.
  virtual const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const =0;

  //! Informations d'initialisation du runtime pour les accélérateurs
  virtual const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const =0;

  //! Fabrique principale
  virtual IMainFactory* mainFactory() const =0;

  //! Application
  virtual IApplication* application() const =0;

 public:
  
  /*!
   * \brief Indique que certains objets sont gérés via un ramasse miette.
   */
  virtual bool hasGarbageCollector() const =0;

 public:

  //! Liste des fabriques de service enregistrées
  virtual ServiceFactoryInfoCollection registeredServiceFactoryInfos() =0;

  //! Liste des fabriques de module enregistrées
  virtual ModuleFactoryInfoCollection registeredModuleFactoryInfos() =0;

 public:

  virtual void setDirectExecuteFunctor(IDirectSubDomainExecuteFunctor* f) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
