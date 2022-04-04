// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopEntryPointInfo.h                                    (C) 2000-2016 */
/*                                                                           */
/* Informations sur un point d'entrée de la boucle en temps.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TIMELOOPENTRYPOINTINFO_H
#define ARCANE_TIMELOOPENTRYPOINTINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Infos d'un point d'entrée d'une boucle en temps.
 */
class TimeLoopEntryPointInfo
{
 public:

  TimeLoopEntryPointInfo() {}
  TimeLoopEntryPointInfo(const String& aname)
  : m_name(aname) { }
  TimeLoopEntryPointInfo(const String& aname,const StringList& modules_depend)
  : m_name(aname), m_modules_depend(modules_depend)
  { }

 public:

  const String& name() const
  { return m_name; }
  const StringList& modulesDepend() const
  { return m_modules_depend; }

  bool operator==(const TimeLoopEntryPointInfo& rhs)
  {
    if (m_name!=rhs.m_name)
      return false;
    if (m_modules_depend.count()!=rhs.m_modules_depend.count())
      return false;
    for( Integer i=0, is=m_modules_depend.count(); i<is; ++i )
      if (m_modules_depend[i]!=rhs.m_modules_depend[i])
        return false;
    return true;
  }

 private:

  String m_name;
  StringList m_modules_depend;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

