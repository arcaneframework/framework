// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IArcaneMain.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of the ArcaneMain class.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IARCANEMAIN_H
#define ARCANE_CORE_IARCANEMAIN_H
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
 * \brief Interface of the code management class.
 *
 This virtual class is used for the creation and initialization of instances
 of code managers. It also controls the execution of a case.
 
 An instance of this class is created via the method
 IMainFactory::createArcaneMain(), called by
 IMainFactory::arcaneMain().

 * The implementation must at least take into account the following aspects.
 * - Analyze the command line.
 * - Create an instance of a supervisor (IMainFactory::createSuperMng()),
 * build it (ISuperMng::build()) and initialize it (ISuperMng::initialize()).
 * - Create an instance of the module loader (IMainFactory::createModuleLoader())
 */
class ARCANE_CORE_EXPORT IArcaneMain
{
 public:

  //! Releases resources.
  virtual ~IArcaneMain() {}

 public:

  /*!
   * Retrieves the global instance.
   *
   * \warning The global instance is only available during the call to
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
   * \brief Constructs the class members.
   * The instance is not usable until this method has been
   * called. This method must be called before initialize().
   * \warning This method must only be called once.
   */
  virtual void build() = 0;

  /*!
   * \brief Initializes the instance.
   * The instance is not usable until this method has been
   * called.
   * \warning This method must only be called once.
   */
  virtual void initialize() = 0;

 public:

  /*! \brief Parses arguments.
   *
   * Recognized arguments must be removed from the list.
   *
   * \retval true if execution must stop,
   * \retval false if it continues normally
   */
  virtual bool parseArgs(StringList args) = 0;

  /*! \brief Starts execution.
   * This method only returns when the program exits.
   * \return the Arcane return code, 0 if everything is okay.
   */
  virtual int execute() = 0;

  //! Performs the last operations before instance destruction
  virtual void finalize() = 0;

  //! Execution error code
  virtual int errorCode() const = 0;

  //! Sets the return code
  virtual void setErrorCode(int errcode) = 0;

  //! Performs an abort.
  virtual void doAbort() = 0;

 public:

  //! Executable information
  virtual const ApplicationInfo& applicationInfo() const = 0;

  //! Information to build the IApplication instance.
  virtual const ApplicationBuildInfo& applicationBuildInfo() const = 0;

  //! .Net runtime initialization information.
  virtual const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const = 0;

  //! Runtime initialization information for accelerators
  virtual const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const = 0;

  //! Main factory
  virtual IMainFactory* mainFactory() const = 0;

  //! Application
  virtual IApplication* application() const = 0;

 public:

  /*!
   * \brief Indicates that certain objects are managed via a garbage collector.
   */
  virtual bool hasGarbageCollector() const = 0;

 public:

  //! List of registered service factories
  virtual ServiceFactoryInfoCollection registeredServiceFactoryInfos() = 0;

  //! List of registered module factories
  virtual ModuleFactoryInfoCollection registeredModuleFactoryInfos() = 0;

 public:

  virtual void setDirectExecuteFunctor(IDirectSubDomainExecuteFunctor* f) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
