// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreApplicationBuildInfo.h                               (C) 2000-2026 */
/*                                                                           */
/* Information for building an instance of IApplication.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARCCOREAPPLICATIONBUILDINFO_H
#define ARCCORE_COMMON_ARCCOREAPPLICATIONBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ArccoreApplicationBuildInfoImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for initializing an application.
 */
class ARCCORE_COMMON_EXPORT ArccoreApplicationBuildInfo
{
 public:

  ArccoreApplicationBuildInfo();
  ArccoreApplicationBuildInfo(const ArccoreApplicationBuildInfo& rhs);
  ArccoreApplicationBuildInfo& operator=(const ArccoreApplicationBuildInfo& rhs);
  virtual ~ArccoreApplicationBuildInfo();

 public:

  void setTaskImplementationService(const String& name);
  void setTaskImplementationServices(const StringList& names);
  StringList taskImplementationServices() const;

  void setThreadImplementationService(const String& name);
  void setThreadImplementationServices(const StringList& names);
  StringList threadImplementationServices() const;

  Int32 nbTaskThread() const;
  void setNbTaskThread(Integer v);

 public:

  void addParameter(const String& name, const String& value);
  /*!
   * \brief Parses the arguments in \a args.
   *
   * Only arguments of the style *-A,x=b,y=c* are retrieved.
   * The setDefaultValues() method is called at the end of this
   * method.
   */
  void parseArgumentsAndSetDefaultsValues(const CommandLineArguments& args);
  virtual void setDefaultValues();
  virtual void setDefaultServices();

 protected:

  ArccoreApplicationBuildInfoImpl* m_core = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
