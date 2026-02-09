// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreApplicationBuildInfo.h                               (C) 2000-2026 */
/*                                                                           */
/* Informations pour construire une instance de IApplication.                */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour initialiser une application.
 */
class ARCCORE_COMMON_EXPORT ApplicationCoreBuildInfo
{
  class CoreImpl;

 public:

  ApplicationCoreBuildInfo();
  ApplicationCoreBuildInfo(const ApplicationCoreBuildInfo& rhs);
  ~ApplicationCoreBuildInfo();
  ApplicationCoreBuildInfo& operator=(const ApplicationCoreBuildInfo& rhs);

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
   * \brief Analyse les arguments de \a args.
   *
   * On ne récupère que les arguments du style *-A,x=b,y=c*.
   * La méthode setDefaultValues() est appelée à la fin de cette
   * méthode.
   */
  void parseArgumentsAndSetDefaultsValues(const CommandLineArguments& args);
  virtual void setDefaultValues();
  virtual void setDefaultServices();

 protected:

  CoreImpl* m_core = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

