// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleMaster.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface of the Master module.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMODULEMASTER_H
#define ARCANE_CORE_IMODULEMASTER_H
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
 * \internal
 * \brief Interface of the main module.
 * 
 * The main module is the module framing the different actions of the entry points.
 * See the \a ModuleMaster implementation for more details.
 */
class ARCANE_CORE_EXPORT IModuleMaster
{
 public:

  //! Destructor.
  /*! Frees the resources */
  virtual ~IModuleMaster() {}

 public:

  //! Creation of an instance of IModuleMaster
  /*! Currently implemented in \a ModuleMaster */
  static IModuleMaster* createDefault(const ModuleBuildInfo&);

 public:

  //! Returns the options of this module
  virtual CaseOptionsMain* caseoptions() = 0;

  //! Conversion to standard module
  /*! The success of the conversion is linked to the implementation of \a IModuleMaster as \a IModule */
  virtual IModule* toModule() = 0;

  //! Access to 'common' variables shared between all services and modules
  virtual CommonVariables* commonVariables() = 0;

  //! Adds the time loop service
  virtual void addTimeLoopService(ITimeLoopService* tls) = 0;

  /*!
   * \brief Outputs the standard curves.
   *
   * This call adds the standard curves to the ITimeHistoryMng
   * (such as CPUTime, ElapsedTime, TotalMemory, ...) for the current iteration. 
   * By default, if this function is not called, the
   * outputs occur at the end of the iteration.
   */
  virtual void dumpStandardCurves() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
