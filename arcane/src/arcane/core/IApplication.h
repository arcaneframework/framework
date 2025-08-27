// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IApplication.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface de l'application.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IAPPLICATION_H
#define ARCANE_CORE_IAPPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfo;
class IMainFactory;
class IArcaneMain;
class IRessourceMng;
class IIOMng;
class XmlNode;
class ICodeService;
class IParallelMng;
class IParallelSuperMng;
class ISession;
class IDataFactory;
class IPhysicalUnitSystemService;
class ITraceMngPolicy;
class IConfigurationMng;
class ApplicationBuildInfo;
class DotNetRuntimeInitialisationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de l'application.
 *
 Cette classe renferme les informations sur la configuration de l'exécutable.
 
 Il n'existe qu'une seule instance de cette classe par processus (singleton).
 */
class ARCANE_CORE_EXPORT IApplication
: public IBase
{
 public:

  //! Gestionnaire superviseur du parallélisme
  virtual IParallelSuperMng* parallelSuperMng() =0;

  //! Gestionnaire un superviseur séquentiel du parallélisme
  virtual IParallelSuperMng* sequentialParallelSuperMng() =0;

  //! Gestionnaire des entrées/sorties.
  virtual IIOMng* ioMng() =0;

  //! Gestionnaire des configurations d'exécution
  virtual IConfigurationMng* configurationMng() const =0;

  //! Fabrique de donnée
  ARCCORE_DEPRECATED_2021("Use dataFactoryMng() instead")
  virtual IDataFactory* dataFactory() =0;

  //! Fabrique de donnée
  virtual IDataFactoryMng* dataFactoryMng() const =0;

  //! Informations sur l'exécutable
  virtual const ApplicationInfo& applicationInfo() const =0;

  //! Informations sur les paramètres de construction de l'instance
  virtual const ApplicationBuildInfo& applicationBuildInfo() const =0;

  //! Informations d'initialisation du runtime '.Net'.
  virtual const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const =0;

  //! Informations d'initialisation du runtime pour les accélérateurs
  virtual const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const =0;

  //! Numéro de version de l'application
  virtual String versionStr() const =0;

  //! Numéro de version principal (sans la béta) de l'application
  virtual String mainVersionStr() const =0;

  //! Numéro de version majeur et mineure sous la forme M.m
  virtual String majorAndMinorVersionStr() const =0;

  //! Informations sur les options de compilation de l'application
  virtual String targetinfoStr() const =0;

  //! Nom du code
  virtual String codeName() const =0;

  //! Nom de l'application
  virtual String applicationName() const =0;

  //! Nom de l'utilisateur
  virtual String userName() const =0;

  /*
   * \brief Contenu du fichier Xml de configuration du code.
   */
  virtual ByteConstSpan configBuffer() const =0;

  /*
   * \brief Contenu du fichier Xml de configuration utilisateur
   */
  virtual ByteConstSpan userConfigBuffer() const =0;

  //! Chemin du répertoire des configurations utilisateur
  virtual String userConfigPath() const =0;

  //! Ajoute la session \a session
  virtual void addSession(ISession* session) =0;

  //! Supprime la session \a session
  virtual void removeSession(ISession* session) =0;

  //! Liste des sessions
  virtual SessionCollection sessions() =0;

  //! Manufacture principale.
  virtual IMainFactory* mainFactory() const =0;

  //! Liste des informations sur les fabriques des modules
  virtual ModuleFactoryInfoCollection moduleFactoryInfos() =0;

  //! Liste des fabriques de service.
  virtual ServiceFactory2Collection serviceFactories2() =0;

  /*!
   * \brief Retourne le chargeur de cas correspondant au fichier
   * donné par \a file_name.
   */
  virtual Ref<ICodeService> getCodeService(const String& file_name) =0;

  //! Indique que certains objets sont gérés via un ramasse miette.
  virtual bool hasGarbageCollector() const =0;

  //! Service gérant les systèmes d'unités physiques
  virtual IPhysicalUnitSystemService* getPhysicalUnitSystemService() =0;

  //! Politique de configuration d'un gestionnaire de trace.
  virtual ITraceMngPolicy* getTraceMngPolicy() =0;

  /*!
   * \brief Créé et initialise une instance de ITraceMng.
   *
   * L'instance créée est initialisée suivant la politique spécifiée
   * par getTraceMngPolicy().
   * Si les sorties fichiers sont activées, l'instance créé sortira
   * ses informations dans un fichier suffixé par \a file_suffix.
   *
   * Les propriétés de verbosité de l'instance créée sont héritées de
   * \a parent_trace s'il n'est pas nul.
   */
  virtual ITraceMng* createAndInitializeTraceMng(ITraceMng* parent_trace,
                                                 const String& file_suffix) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
