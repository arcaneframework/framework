// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationInfo.h                                           (C) 2000-2024 */
/*                                                                           */
/* Informations sur une application.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_APPLICATIONINFO_H
#define ARCANE_UTILS_APPLICATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfoPrivate;
class CommandLineArguments;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une application.
 */
class ARCANE_UTILS_EXPORT ApplicationInfo
{
 public:

  ApplicationInfo();
  ApplicationInfo(int* argc,char*** argv,const String& name,const VersionInfo& version);
  ApplicationInfo(const StringList& args,const String& name,const VersionInfo& version);
  ApplicationInfo(const CommandLineArguments& args,const String& name,const VersionInfo& version);
  ApplicationInfo(const ApplicationInfo& rhs);
  ~ApplicationInfo();
  ApplicationInfo& operator=(const ApplicationInfo& rhs);

 public:
	
  //! Nom de l'application
  const String& applicationName() const;
  //! Numéro de version
  ARCCORE_DEPRECATED_2020("use codeVersion() instead")
  const VersionInfo& version() const { return codeVersion(); }
  //! Numéro de version
  const VersionInfo& codeVersion() const;
  //! Retourne le chemin où se trouve les fichiers de données dépendant de l'OS
  const String& dataOsDir() const;
  //! Retourne le chemin où se trouve les fichiers de données.
  const String& dataDir() const;

  //! Retourne le numéro de version majeure de l'application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionMajor() instead")
  int versionMajor() const;
  //! Retourne le numéro de version mineure de l'application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionMinor() instead")
  int versionMinor() const;
  //! Retourne le numéro de version patch de l'application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionPatch() instead")
  int versionPatch() const;

  //! Retourne \a true si on s'exécute en mode debug.
  bool isDebug() const;

  //! Retourne le nom du code de calcul lié l'application
  const String& codeName() const;
  //! Retourne le nom complet de la cible
  const String& targetFullName() const;

  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgc() instead")
  int* commandLineArgc() const;
  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgv() instead")
  char*** commandLineArgv() const;

  //! Remplit \a args avec les arguments de la ligne de commande.
  void args(StringList& args) const;

  //! Arguments de la ligne de commande
  const CommandLineArguments& commandLineArguments() const;

 public:

  /*!
   * \brief Ajoute la bibliothèque \a lib_name à la liste des bibliothèques
   * chargées dynamiquements.
   *
   * \a lib_name doit être un nom, sans chemin et sans extension. Par exemple,
   * \c my_lib est valide mais pas \c libtoto.so, ni \c /tmp/toto.
   */
  void addDynamicLibrary(const String& lib_name);

  //! Liste des bibliothèques dynamiques.
  StringCollection dynamicLibrariesName() const;

 public:

  //! Positionne le chemin où se trouve les fichiers de données dépendant de l'OS
  void setDataOsDir(const String& v);
  //! Positionne le chemin où se trouve les fichiers de données.
  void setDataDir(const String& v);
  //! Positionne le numéro de version du code
  ARCCORE_DEPRECATED_2020("use setCodeVersion() instead")
  void setVersionInfo(const VersionInfo& version_info)
  { setCodeVersion(version_info); }
  //! Positionne le nom de l'application
  void setApplicationName(const String& v);
  //! Positionne le numéro de version
  void setCodeVersion(const VersionInfo& version_info);
  //! Positionne le nom du code
  void setCodeName(const String& code_name);
  /*!
   * \brief Positionne les arguments de la ligne de commande.
   *
   * L'appel à cette méthode modifie les valeurs de \a m_argv et \a m_argc.
   */
  void setCommandLineArguments(const CommandLineArguments& args);
  //! Positionne l'état de débug.
  void setIsDebug(bool v);

 public:

  //! Positionne le contenu du fichier de configuration de l'application
  void setRuntimeConfigFileContent(ByteConstSpan content);
  //! Contenu du fichier de configuration de l'application
  ByteConstSpan runtimeConfigFileContent() const;

 public:

  /*!
   * \brief Ajoute un paramètre Arcane à la ligne de commande.
   * \sa ParameterList::addParameterLine().
   */
  void addParameterLine(const String& line);

 public:

  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgc() instead")
  int* m_argc; //!< Nombre d'arguments de la ligne de commande
  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgv() instead")
  char*** m_argv; //!< Tableau des arguments de la ligne de commande

 private:

  ApplicationInfoPrivate* m_p;

 private:
  
  void _init(const String& name);
  void _setArgs();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

