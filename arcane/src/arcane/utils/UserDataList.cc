// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UserDataList.cc                                             (C) 2000-2012 */
/*                                                                           */
/* Gère une liste de données utilisateurs.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/UserDataList.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/IUserData.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UserDataList::Impl
{
 public:
  typedef std::map<String,IUserData*> MapType;
  MapType m_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UserDataList::
UserDataList()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UserDataList::
~UserDataList()
{
  clear();
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserDataList::
clear()
{
  Impl::MapType::const_iterator begin = m_p->m_list.begin();
  Impl::MapType::const_iterator end = m_p->m_list.end();
  for( ; begin!=end; ++begin )
    begin->second->notifyDetach();
  m_p->m_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserDataList::
setData(const String& name,IUserData* ud)
{
  Impl::MapType::const_iterator i = m_p->m_list.find(name);
  if (i!=m_p->m_list.end())
    throw ArgumentException(A_FUNCINFO,String::format("key '{0}' already exists",name));
  m_p->m_list.insert(std::make_pair(name,ud));
  ud->notifyAttach();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IUserData* UserDataList::
data(const String& name,bool allow_null) const
{
  Impl::MapType::const_iterator i = m_p->m_list.find(name);
  if (i==m_p->m_list.end()){
    if (allow_null)
      return 0;
    throw ArgumentException(A_FUNCINFO,String::format("key '{0}' not found",name));
  }
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserDataList::
removeData(const String& name,bool allow_null)
{
  Impl::MapType::iterator i = m_p->m_list.find(name);
  if (i==m_p->m_list.end()){
    if (allow_null)
      return;
    throw ArgumentException(A_FUNCINFO,String::format("key '{0}' not found",name));
  }
  i->second->notifyDetach();
  m_p->m_list.erase(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
