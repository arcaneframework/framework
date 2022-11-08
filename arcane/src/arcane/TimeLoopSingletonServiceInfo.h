// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopSingletonServiceInfo.h                              (C) 2000-2022 */
/*                                                                           */
/* Infos d'un service singleton d'une boucle en temps.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TIMELOOPSINGLETONSERVICEINFO_H
#define ARCANE_TIMELOOPSINGLETONSERVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Infos d'un service singleton d'une boucle en temps.
 */
class TimeLoopSingletonServiceInfo
{
 public:

  TimeLoopSingletonServiceInfo()
  : m_is_required(false){}
  TimeLoopSingletonServiceInfo(const String& name,bool is_required)
  : m_name(name), m_is_required(is_required){}

 public:

  const String& name() const { return m_name; }
  bool isRequired() const { return m_is_required; }
  bool operator==(const TimeLoopSingletonServiceInfo& rhs) const
  {
    if (m_name!=rhs.m_name)
      return false;
    return m_is_required!=rhs.m_is_required;
  }

 private:

  String m_name;
  bool m_is_required;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

