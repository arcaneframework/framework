// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoop.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface of a time loop.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMELOOP_H
#define ARCANE_CORE_ITIMELOOP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Module
 * \brief Interface of a time loop.
 */
class ARCANE_CORE_EXPORT ITimeLoop
{
 public:

  /*! @name Call point
    Where the entry point is used.
   */
  //@{
  //! called during the calculation loop
  static const char* WComputeLoop;
  //! called when reading the dataset
  static const char* WBuild;
  //! called during initialization, initialization of a restart, or a new case
  static const char* WInit;
  //! called to restore variables during a rollback
  static const char* WRestore;
  //! called after a mesh change
  static const char* WOnMeshChanged;
  //! called after mesh refinement
  static const char* WOnMeshRefinement;
  //! called upon termination of the code.
  static const char* WExit;
  //@}

 public:

  virtual ~ITimeLoop() = default; //!< Frees resources.

 public:

  //! Constructs the time loop
  virtual void build() = 0;

 public:

  //! Application
  virtual IApplication* application() const = 0;

 public:

  //! Name of the time loop
  virtual String name() const = 0;

  //! Title of the time loop
  virtual String title() const = 0;

  //! Sets the title of the time loop
  virtual void setTitle(const String&) = 0;

  //! Description of the time loop
  virtual String description() const = 0;

  //! Sets the description of the time loop
  virtual void setDescription(const String&) = 0;

  //! List of names of required modules.
  virtual StringCollection requiredModulesName() const = 0;

  //! Sets the list of required modules.
  virtual void setRequiredModulesName(const StringCollection&) = 0;

  //! List of names of optional modules.
  virtual StringCollection optionalModulesName() const = 0;

  //! Sets the list of optional modules.
  virtual void setOptionalModulesName(const StringCollection&) = 0;

  //! List of names of entry points for the call point \a where.
  virtual TimeLoopEntryPointInfoCollection entryPoints(const String& where) const = 0;

  //! Sets the list of names of entry points for the call point \a where
  virtual void setEntryPoints(const String& where, const TimeLoopEntryPointInfoCollection&) = 0;

  //! List of user classes associated with the time loop.
  virtual StringCollection userClasses() const = 0;

  //! Returns the list of classes associated with the time loop.
  virtual void setUserClasses(const StringCollection&) = 0;

  //! List of singleton services
  virtual TimeLoopSingletonServiceInfoCollection singletonServices() const = 0;

  //! Sets the list of singleton services.
  virtual void setSingletonServices(const TimeLoopSingletonServiceInfoCollection& c) = 0;

  //! Configuration options
  virtual IConfiguration* configuration() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
