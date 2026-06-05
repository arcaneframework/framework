// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSimpleExecutor.h                                      (C) 2000-2026 */
/*                                                                           */
/* Class for executing code directly via Arcane.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNALINFOSDUMPER_H
#define ARCANE_IMPL_INTERNALINFOSDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class ApplicationInfo;
class ApplicationBuildInfo;
class CommandLineArguments;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for directly executing code without
 * going through the time loop.
 *
 * Only one instance of this class must exist at any given time.
 *
 * Instances of this class use the value of
 * ArcaneMain::defaultApplicationInfo() to initialize themselves and notably
 * retrieve the command line arguments.
 *
 * The initialize() method must be called before calling other
 * methods such as createSubDomain(). It is possible to modify the
 * application creation parameters by modifying the values
 * of the instance returned by applicationBuildInfo().
 */
class ARCANE_IMPL_EXPORT ArcaneSimpleExecutor
{
  class Impl;

 public:

  ArcaneSimpleExecutor();
  ArcaneSimpleExecutor(const ArcaneSimpleExecutor&) = delete;
  ~ArcaneSimpleExecutor() noexcept(false);
  const ArcaneSimpleExecutor& operator=(const ArcaneSimpleExecutor&) = delete;

 public:

  ApplicationBuildInfo& applicationBuildInfo();
  const ApplicationBuildInfo& applicationBuildInfo() const;

  int initialize();
  ISubDomain* createSubDomain(const String& case_file_name);
  int runCode(IFunctor* f);

 private:

  Impl* m_p;

 private:

  void _checkInit();
  void _setDefaultVerbosityLevel(Integer level);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
