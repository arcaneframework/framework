// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationBuildInfo.h                                      (C) 2000-2026 */
/*                                                                           */
/* Information for constructing an instance of IApplication.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_APPLICATIONBUILDINFO_H
#define ARCANE_UTILS_APPLICATIONBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/common/ArccoreApplicationBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CaseDatasetSource;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for constructing an instance of IApplication.
 */
class ARCANE_CORE_EXPORT ApplicationBuildInfo
: public ArccoreApplicationBuildInfo
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
   * \brief Sets the code configuration file.
   * \sa configFileName().
   */
  void setConfigFileName(const String& name);

  /*!
   * \brief Name of the code configuration file.
   *
   * By default, the value is an empty string ("").
   * In this case, %Arcane searches for a file whose name
   * is codeName() followed by the `.config` extension.
   *
   * If the value is null, then no configuration file is loaded.
   */
  String configFileName() const;

  /*!
   * \brief Sets the message verbosity level
   * on standard output.
   */
  void setOutputLevel(Int32 v);
  Int32 outputLevel() const;

  /*!
   * \brief Sets the message verbosity level
   * of reduced listing files.
   */
  void setVerbosityLevel(Int32 v);
  Int32 verbosityLevel() const;

  Int32 minimalVerbosityLevel() const;
  void setMinimalVerbosityLevel(Int32 v);

  bool isMasterHasOutputFile() const;
  void setIsMasterHasOutputFile(bool v);

  /*!
   * \brief Sets the directory containing the various
   * simulation outputs.
   *
   * These outputs include the summary, profiling traces,
   * listing outputs, ...
   */
  void setOutputDirectory(const String& name);
  String outputDirectory() const;

 public:

  //! Sets the application name
  void setApplicationName(const String& v);
  //! Application name
  String applicationName() const;

  //! Sets the code version
  void setCodeVersion(const VersionInfo& version_info);
  //! Version number
  VersionInfo codeVersion() const;

  //! Sets the code name
  void setCodeName(const String& code_name);
  //! Returns the code name.
  String codeName() const;

  //! Strategy for binding task threads
  String threadBindingStrategy() const;

  //! Sets the strategy for binding task threads
  void threadBindingStrategy(const String& v);

  //! Dataset source
  CaseDatasetSource& caseDatasetSource();
  //! Dataset source
  const CaseDatasetSource& caseDatasetSource() const;

 public:

  /*!
   * \brief Default message passing manager name.
   * Must only be modified by Arcane.
   */
  void internalSetDefaultMessagePassingService(const String& name);
  String internalDefaultMessagePassingService() const;

 public:

  /*!
   * \brief Adds the library \a lib_name to the list of
   * dynamically loaded libraries.
   *
   * \a lib_name must be a name, without path or extension. For example,
   * \c my_lib is valid but not \c libtoto.so, nor \c /tmp/toto.
   */
  void addDynamicLibrary(const String& lib_name);

 public:

  /*!
   * \brief Parses the arguments in \a args.
   *
   * Only arguments of the style *-A,x=b,y=c* are retrieved.
   * The setDefaultValues() method is called at the end of this
   * method.
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
