// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationBuildInfo.h                                      (C) 2000-2021 */
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
 * \brief Informations pour construire une instance de IApplication.
 */
class ARCANE_CORE_EXPORT ApplicationBuildInfo
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

  void setTaskImplementationService(const String& name);
  void setTaskImplementationServices(const StringList& names);
  StringList taskImplementationServices() const;

  void setThreadImplementationService(const String& name);
  void setThreadImplementationServices(const StringList& names);
  StringList threadImplementationServices() const;

  Int32 nbTaskThread() const;
  void setNbTaskThread(Integer v);

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

  Int32 outputLevel() const;
  void setOutputLevel(Int32 v);

  Int32 verbosityLevel() const;
  void setVerbosityLevel(Int32 v);

  Int32 minimalVerbosityLevel() const;
  void setMinimalVerbosityLevel(Int32 v);

  bool isMasterHasOutputFile() const;
  void setIsMasterHasOutputFile(bool v);

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

  //! Source du jeu de données
  CaseDatasetSource& caseDatasetSource();
  //! Source du jeu de données
  const CaseDatasetSource& caseDatasetSource() const;

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

  void addParameter(const String& name,const String& value);
  /*!
   * \brief Analyse les arguments de \a args.
   *
   * On ne récupère que les arguments du style *-A,x=b,y=c*.
   * La méthode setDefaultValues() est appelée à la fin de cette
   * méthode.
   */
  void parseArguments(const CommandLineArguments& args);

 public:

  ApplicationInfo& _internalApplicationInfo();
  const ApplicationInfo& _internalApplicationInfo() const;

 public:

  void setDefaultValues();
  void setDefaultServices();

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

