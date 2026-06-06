// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IApplication.h                                              (C) 2000-2025 */
/*                                                                           */
/* Application interface.                                                    */
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
 * \brief Application interface.
 *
 This class contains information about the executable configuration.
 
 There is only one instance of this class per process (singleton).
 */
class ARCANE_CORE_EXPORT IApplication
: public IBase
{
 public:

  //! Supervisory parallelism manager
  virtual IParallelSuperMng* parallelSuperMng() =0;

  //! Sequential parallelism supervisor manager
  virtual IParallelSuperMng* sequentialParallelSuperMng() =0;

  //! Input/output manager.
  virtual IIOMng* ioMng() =0;

  //! Execution configuration manager
  virtual IConfigurationMng* configurationMng() const =0;

  //! Data factory
  ARCCORE_DEPRECATED_2021("Use dataFactoryMng() instead")
  virtual IDataFactory* dataFactory() =0;

  //! Data factory
  virtual IDataFactoryMng* dataFactoryMng() const =0;

  //! Executable information
  virtual const ApplicationInfo& applicationInfo() const =0;

  //! Instance build parameter information
  virtual const ApplicationBuildInfo& applicationBuildInfo() const =0;

  //! '.Net' runtime initialization information.
  virtual const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const =0;

  //! Runtime initialization information for accelerators
  virtual const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const =0;

  //! Application version number
  virtual String versionStr() const =0;

  //! Main application version number (without beta)
  virtual String mainVersionStr() const =0;

  //! Major and minor version number in M.m format
  virtual String majorAndMinorVersionStr() const =0;

  //! Application compilation options information
  virtual String targetinfoStr() const =0;

  //! Code name
  virtual String codeName() const =0;

  //! Application name
  virtual String applicationName() const =0;

  //! User name
  virtual String userName() const =0;

  /*
   * \brief Content of the code configuration Xml file.
   */
  virtual ByteConstSpan configBuffer() const =0;

  /*
   * \brief Content of the user configuration Xml file
   */
  virtual ByteConstSpan userConfigBuffer() const =0;

  //! User configuration directory path
  virtual String userConfigPath() const =0;

  //! Adds the session \a session
  virtual void addSession(ISession* session) =0;

  //! Removes the session \a session
  virtual void removeSession(ISession* session) =0;

  //! List of sessions
  virtual SessionCollection sessions() =0;

  //! Main factory.
  virtual IMainFactory* mainFactory() const =0;

  //! List of module factory information
  virtual ModuleFactoryInfoCollection moduleFactoryInfos() =0;

  //! List of service factories.
  virtual ServiceFactory2Collection serviceFactories2() =0;

  /*!
   * \brief Returns the case loader corresponding to the file
   * given by \a file_name.
   */
  virtual Ref<ICodeService> getCodeService(const String& file_name) =0;

  //! Indicates that certain objects are managed via a garbage collector.
  virtual bool hasGarbageCollector() const =0;

  //! Service managing physical unit systems
  virtual IPhysicalUnitSystemService* getPhysicalUnitSystemService() =0;

  //! Trace manager configuration policy.
  virtual ITraceMngPolicy* getTraceMngPolicy() =0;

  /*!
   * \brief Creates and initializes an instance of ITraceMng.
   *
   * The created instance is initialized according to the policy specified
   * by getTraceMngPolicy().
   * If file outputs are enabled, the created instance will output
   * its information into a file suffixed by \a file_suffix.
   *
   * The verbosity properties of the created instance are inherited from
   * \a parent_trace if it is not null.
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
