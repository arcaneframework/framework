// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationInfo.h                                           (C) 2000-2024 */
/*                                                                           */
/* Application information.                                                  */
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
 * \brief Application information.
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
	
  //! Application name
  const String& applicationName() const;
  //! Version number
  ARCCORE_DEPRECATED_2020("use codeVersion() instead")
  const VersionInfo& version() const { return codeVersion(); }
  //! Version number
  const VersionInfo& codeVersion() const;
  //! Returns the path where OS-dependent data files are located
  const String& dataOsDir() const;
  //! Returns the path where data files are located.
  const String& dataDir() const;

  //! Returns the major version number of the application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionMajor() instead")
  int versionMajor() const;
  //! Returns the minor version number of the application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionMinor() instead")
  int versionMinor() const;
  //! Returns the patch version number of the application
  ARCCORE_DEPRECATED_2020("use codeVersion().versionPatch() instead")
  int versionPatch() const;

  //! Returns \a true if running in debug mode.
  bool isDebug() const;

  //! Returns the name of the calculation code linked to the application
  const String& codeName() const;
  //! Returns the full target name
  const String& targetFullName() const;

  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgc() instead")
  int* commandLineArgc() const;
  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgv() instead")
  char*** commandLineArgv() const;

  //! Fills \a args with command line arguments.
  void args(StringList& args) const;

  //! Command line arguments
  const CommandLineArguments& commandLineArguments() const;

 public:

  /*!
   * \brief Adds the library \a lib_name to the list of dynamically loaded libraries.
   *
   * \a lib_name must be a name, without path and without extension. For example,
   * \c my_lib is valid but not \c libtoto.so, nor \c /tmp/toto.
   */
  void addDynamicLibrary(const String& lib_name);

  //! List of dynamic libraries.
  StringCollection dynamicLibrariesName() const;

 public:

  //! Sets the path where OS-dependent data files are located
  void setDataOsDir(const String& v);
  //! Sets the path where data files are located.
  void setDataDir(const String& v);
  //! Sets the code version number
  ARCCORE_DEPRECATED_2020("use setCodeVersion() instead")
  void setVersionInfo(const VersionInfo& version_info)
  { setCodeVersion(version_info); }
  //! Sets the application name
  void setApplicationName(const String& v);
  //! Sets the version number
  void setCodeVersion(const VersionInfo& version_info);
  //! Sets the code name
  void setCodeName(const String& code_name);
  /*!
   * \brief Sets the command line arguments.
   *
   * Calling this method modifies the values of \a m_argv and \a m_argc.
   */
  void setCommandLineArguments(const CommandLineArguments& args);
  //! Sets the debug state.
  void setIsDebug(bool v);

 public:

  //! Sets the application configuration file content
  void setRuntimeConfigFileContent(ByteConstSpan content);
  //! Application configuration file content
  ByteConstSpan runtimeConfigFileContent() const;

 public:

  /*!
   * \brief Adds an Arcane parameter to the command line.
   * \sa ParameterList::addParameterLine().
   */
  void addParameterLine(const String& line);

 public:

  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgc() instead")
  int* m_argc; //!< Number of command line arguments
  ARCCORE_DEPRECATED_2019("Use commandLineArguments().commandLineArgv() instead")
  char*** m_argv; //!< Array of command line arguments

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
