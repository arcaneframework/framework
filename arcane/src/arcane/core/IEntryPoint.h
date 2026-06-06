// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEntryPoint.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a module entry point.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IENTRYPOINT_H
#define ARCANE_CORE_IENTRYPOINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Timer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IModule;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a module entry point.
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT IEntryPoint
{
 public:

  /*! @name Call point
    Where the entry point is used.
   */
  //@{
  //! called during the calculation loop
  static const char* const WComputeLoop;
  //! called for module construction
  static const char* const WBuild;
  //! called during initialization
  static const char* const WInit;
  //! called during continuation initialization
  static const char* const WContinueInit;
  //! called during new case initialization
  static const char* const WStartInit;
  //! called to restore variables during a rollback
  static const char* const WRestore;
  //! called after a mesh change
  static const char* const WOnMeshChanged;
  //! called after mesh refinement
  static const char* const WOnMeshRefinement;
  //!< called upon code termination.
  static const char* const WExit;
  //@}

  /*!
   * \brief Properties of an entry point.
   */
  enum
  {
    PNone = 0, //!< No properties
    /*!
     * \brief Automatically loaded at the beginning.
     * This means that a module possessing an entry point with this
     * property will always be loaded, and the entry point will be added
     * to the list of entry points executing at the beginning of the time loop.
     */
    PAutoLoadBegin = 1,
    /*!
     * \brief Automatically loaded at the end.
     * This means that a module possessing an entry point with this
     * property will always be loaded, and the entry point will be added
     * to the list of entry points executing at the end of the time loop.
     */
    PAutoLoadEnd = 2
  };

 public:

  virtual ~IEntryPoint() = default; //!< Releases resources

 public:

  //! Returns the name of the entry point.
  virtual String name() const = 0;

  //! Full name (with the module) of the entry point. This name is unique.
  virtual String fullName() const = 0;

 public:

  //! Returns the main manager
  ARCANE_DEPRECATED_REASON("Y2022: Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() const = 0;

  //! Returns the module associated with the entry point
  virtual IModule* module() const = 0;

  //! Calls the entry point
  virtual void executeEntryPoint() = 0;

  /*!
   * \brief Total CPU consumption spent in this entry point (in milliseconds).
   *
   * \note since version 3.6 of Arcane, this method returns the same value
   * as totalElapsedTime().
   */
  virtual Real totalCPUTime() const = 0;

  /*!
   * \brief CPU consumption of the last iteration (in milliseconds).
   *
   * \note since version 3.6 of Arcane, this method returns the same value
   * as lastElapsedTime().
   */
  virtual Real lastCPUTime() const = 0;

  //! Elapsed execution time (clock time) in this entry point (in milliseconds)
  virtual Real totalElapsedTime() const = 0;

  //! Elapsed execution time (clock time) of the last iteration (in milliseconds).
  virtual Real lastElapsedTime() const = 0;

  /*!
   * \brief Returns totalElapsedTime().
   * \deprecated Use totalElapsedTime() instead
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use totalElapsedTime() instead")
  virtual Real totalTime(Timer::eTimerType) const = 0;

  /*!
   * \brief Returns lastElapsedTime().
   * \deprecated Use lastElapsedTime() instead
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use lastElapsedTime() instead")
  virtual Real lastTime(Timer::eTimerType) const = 0;

  //! Returns the number of times the entry point has been executed
  virtual Integer nbCall() const = 0;

  //! Returns where the entry point is called.
  virtual String where() const = 0;

  //! Returns the properties of the entry point.
  virtual int property() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
