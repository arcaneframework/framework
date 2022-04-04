// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoop.cc                                                 (C) 2000-2014 */
/*                                                                           */
/* Boucle en temps.                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"

#include "arcane/TimeLoop.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/TimeLoopSingletonServiceInfo.h"
#include "arcane/Configuration.h"
#include "arcane/IApplication.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* ITimeLoop::WComputeLoop = "compute-loop";
const char* ITimeLoop::WBuild = "build";
const char* ITimeLoop::WInit = "init";
const char* ITimeLoop::WRestore = "restore";
const char* ITimeLoop::WOnMeshChanged = "on-mesh-changed";
const char* ITimeLoop::WOnMeshRefinement = "on-mesh-refinement";
const char* ITimeLoop::WExit = "exit";

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef List<TimeLoopEntryPointInfo> TimeLoopEntryPointInfoList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Boucle en temps.
 */
class TimeLoopPrivate
{
 public:

  typedef std::map<String,List< TimeLoopEntryPointInfo > > EntryPointInfoMap;

 public:
  
  TimeLoopPrivate(IApplication * mng, const String & name);
  ~TimeLoopPrivate(){ delete m_configuration; }

 public:

  IApplication* m_application; //!< Application
  String m_name; //!< Nom informatique
  String m_title; //!< Titre
  String m_description; //!< Description
  bool m_is_old; //! Vrai si boucle en temps au vieux format
  StringList m_required_modules_name;
  StringList m_optional_modules_name;
  StringList m_user_classes; //!< Liste des classes utilisateurs.
  EntryPointInfoMap m_entry_points;
  List<TimeLoopSingletonServiceInfo> m_singleton_services;
  IConfiguration* m_configuration;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT ITimeLoop*
arcaneCreateTimeLoop(IApplication* app,const String& name)
{
  ITimeLoop * tm = new TimeLoop(app,name);
  tm->build();
  return tm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopPrivate::
TimeLoopPrivate(IApplication* app,const String& name)
: m_application(app)
, m_name(name)
, m_is_old(false)
, m_configuration(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoop::
TimeLoop(IApplication* app,const String& name)
: m_p(new TimeLoopPrivate(app,name))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoop::
~TimeLoop()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
build()
{
  m_p->m_configuration = m_p->m_application->configurationMng()->createConfiguration();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
setRequiredModulesName(const StringCollection& names)
{
  m_p->m_required_modules_name.clone(names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
setOptionalModulesName(const StringCollection& names)
{
  m_p->m_optional_modules_name.clone(names);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopEntryPointInfoCollection TimeLoop::
entryPoints(const String& where) const
{
  TimeLoopPrivate::EntryPointInfoMap::const_iterator it = m_p->m_entry_points.find(where);
  if (it==m_p->m_entry_points.end()){
    return List<TimeLoopEntryPointInfo>();
  }
  return it->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
setEntryPoints(const String& where,
               const TimeLoopEntryPointInfoCollection& calls)
{
  TimeLoopEntryPointInfoList entry_points;
  entry_points.clone(calls);
  m_p->m_entry_points.insert(TimeLoopPrivate::EntryPointInfoMap::value_type(where,entry_points));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
setUserClasses(const StringCollection& user_classes)
{
  m_p->m_user_classes.clone(user_classes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication * TimeLoop::
application() const
{
  return m_p->m_application;
}

String TimeLoop::
name() const
{
  return m_p->m_name;
}

String TimeLoop::
title() const
{
  return m_p->m_title;
}

void TimeLoop::
setTitle(const String& title)
{
  m_p->m_title = title;
}

bool TimeLoop::
isOldFormat() const
{
  return m_p->m_is_old;
}

void TimeLoop::
setOldFormat(bool is_old)
{
  m_p->m_is_old = is_old;
}

String TimeLoop::
description() const
{
  return m_p->m_description;
}

void TimeLoop::
setDescription(const String& description)
{
  m_p->m_description = description;
}

StringCollection TimeLoop::
requiredModulesName() const
{
  return m_p->m_required_modules_name;
}

StringCollection TimeLoop::
optionalModulesName() const
{
  return m_p->m_optional_modules_name;
}

StringCollection TimeLoop::
userClasses() const
{
  return m_p->m_user_classes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopSingletonServiceInfoCollection TimeLoop::
singletonServices() const
{
  return m_p->m_singleton_services;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoop::
setSingletonServices(const TimeLoopSingletonServiceInfoCollection& c)
{
  m_p->m_singleton_services.clone(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IConfiguration* TimeLoop::
configuration()
{
  return m_p->m_configuration;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
