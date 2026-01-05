// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationBuildInfo.h                                      (C) 2000-2026 */
/*                                                                           */
/* Informations pour construire une instance de IApplication.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_APPLICATIONBUILDINFO_H
#define ARCANE_UTILS_APPLICATIONBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CaseDatasetSource;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour initialiser une application.
 */
class ARCANE_CORE_EXPORT ApplicationCoreBuildInfo
{
  class CoreImpl;

 public:

  ApplicationCoreBuildInfo();
  ApplicationCoreBuildInfo(const ApplicationCoreBuildInfo& rhs);
  ~ApplicationCoreBuildInfo();
  ApplicationCoreBuildInfo& operator=(const ApplicationCoreBuildInfo& rhs);

 public:

  void setTaskImplementationService(const String& name);
  void setTaskImplementationServices(const StringList& names);
  StringList taskImplementationServices() const;

  void setThreadImplementationService(const String& name);
  void setThreadImplementationServices(const StringList& names);
  StringList threadImplementationServices() const;

  Int32 nbTaskThread() const;
  void setNbTaskThread(Integer v);

 public:

  void addParameter(const String& name, const String& value);
  /*!
   * \brief Analyse les arguments de \a args.
   *
   * On ne récupère que les arguments du style *-A,x=b,y=c*.
   * La méthode setDefaultValues() est appelée à la fin de cette
   * méthode.
   */
  void parseArgumentsAndSetDefaultsValues(const CommandLineArguments& args);
  virtual void setDefaultValues();
  virtual void setDefaultServices();

 protected:

  CoreImpl* m_core = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour construire une instance de IApplication.
 */
class ARCANE_CORE_EXPORT ApplicationBuildInfo
: public ApplicationCoreBuildInfo
{
  class Impl;

 public:

  ApplicationBuildInfo();
  ApplicationBuildInfo(const ApplicationBuildInfo& rhs);
  ~ApplicationBuildInfo();
  ApplicationBuildInfo& operator=(const ApplicationBuildInfo& rhs);

 public:

  void setMessagePassingService(const String& name);
  String messagePassingService() const;

  Int32 nbSharedMemorySubDomain() const;
  void setNbSharedMemorySubDomain(Int32 v);

  Int32 nbReplicationSubDomain() const;
  void setNbReplicationSubDomain(Int32 v);

  Int32 nbProcessusSubDomain() const;
  void setNbProcessusSubDomain(Int32 v);

  /*!
   * \brief Positionne le fichier de configuration du code.
   * \sa configFileName().
   */
  void setConfigFileName(const String& name);

  /*!
   * \brief Nom du fichier de configuration du code.
   *
   * Par défaut, la valeur est celle de la chaîne vide ("").
   * Dans ce cas, %Arcane recherche un fichier dont le nom
   * est codeName() suivi de l'extension `.config`.
   *
   * Si la valeur est nulle, alors il n'y a pas de fichier de
   * configuration chargé.
   */
  String configFileName() const;

  /*!
   * \brief Positionne le niveau de verbosité des messages
   * sur la sortie standard.
   */
  void setOutputLevel(Int32 v);
  Int32 outputLevel() const;

  /*!
   * \brief Positionne le niveau de verbosité des messages
   * des fichiers listings réduits.
   */
  void setVerbosityLevel(Int32 v);
  Int32 verbosityLevel() const;

  Int32 minimalVerbosityLevel() const;
  void setMinimalVerbosityLevel(Int32 v);

  bool isMasterHasOutputFile() const;
  void setIsMasterHasOutputFile(bool v);

  /*!
   * \brief Positionne le répertoire contenant les différentes sorties
   * de la simulation.
   *
   * Parmi ces sorties on trouve le dépouillement, les traces de profilage,
   * les sorties listings, ...
   */
  void setOutputDirectory(const String& name);
  String outputDirectory() const;

 public:

  //! Positionne le nom de l'application
  void setApplicationName(const String& v);
  //! Nom de l'application
  String applicationName() const;

  //! Positionne le numéro de version du code
  void setCodeVersion(const VersionInfo& version_info);
  //! Numéro de version
  VersionInfo codeVersion() const;

  //! Positionne le nom du code
  void setCodeName(const String& code_name);
  //! Retourne le nom du code.
  String codeName() const;

  //! Stratégie pour punaiser les threads des tâches
  String threadBindingStrategy() const;

  //! Positionne la strategie pour punaiser les threads des tâches
  void threadBindingStrategy(const String& v);

  //! Source du jeu de données
  CaseDatasetSource& caseDatasetSource();
  //! Source du jeu de données
  const CaseDatasetSource& caseDatasetSource() const;

 public:

  /*!
   * \brief Nom du gestionnaire de message par défaut.
   * Ne doit être modifié que par Arcane.
   */
  void internalSetDefaultMessagePassingService(const String& name);
  String internalDefaultMessagePassingService() const;

 public:

  /*!
   * \brief Ajoute la bibliothèque \a lib_name à la liste des bibliothèques
   * chargées dynamiquements.
   *
   * \a lib_name doit être un nom, sans chemin et sans extension. Par exemple,
   * \c my_lib est valide mais pas \c libtoto.so, ni \c /tmp/toto.
   */
  void addDynamicLibrary(const String& lib_name);

 public:

  /*!
   * \brief Analyse les arguments de \a args.
   *
   * On ne récupère que les arguments du style *-A,x=b,y=c*.
   * La méthode setDefaultValues() est appelée à la fin de cette
   * méthode.
   */
  ARCANE_DEPRECATED_REASON("Use parseArgumentsAndSetDefaultsValues() instead")
  void parseArguments(const CommandLineArguments& args)
  {
    parseArgumentsAndSetDefaultsValues(args);
  }

 public:

  ApplicationInfo& _internalApplicationInfo();
  const ApplicationInfo& _internalApplicationInfo() const;

 public:

  void setDefaultValues() override;
  void setDefaultServices() override;

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

