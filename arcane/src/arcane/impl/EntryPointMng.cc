// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EntryPointMng.cc                                            (C) 2000-2007 */
/*                                                                           */
/* Gestionnaire des points d'entrée.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/IEntryPointMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/IModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des points d'entrée.
 */
class EntryPointMng
: public TraceAccessor
, public IEntryPointMng
{
 public:

  EntryPointMng(ISubDomain*);
  ~EntryPointMng();

 public:
  
  virtual void addEntryPoint(IEntryPoint*);
  virtual void dumpList(ostream&);
  virtual IEntryPoint* findEntryPoint(const String& s);
  virtual IEntryPoint* findEntryPoint(const String& module_name,const String& s);
  virtual EntryPointCollection entryPoints() { return m_entry_points; }

 private:

  EntryPointList m_entry_points; //!< Liste des points d'entrée
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IEntryPointMng*
arcaneCreateEntryPointMng(ISubDomain* sd)
{
  return new EntryPointMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPointMng::
EntryPointMng(ISubDomain* sd)
: TraceAccessor(sd->traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntryPointMng::
~EntryPointMng()
{
  for( EntryPointList::Enumerator i(m_entry_points); ++i; )
    delete *i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EntryPointMng::
addEntryPoint(IEntryPoint* v)
{
  log() << " Add an entry point <" << v->module()->name() << "::" << v->name() << ">";
  m_entry_points.add(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EntryPointMng::
dumpList(ostream& o)
{
  o << "** EntryPointMng::dump_list: " << m_entry_points.count();
  o << '\n';
  for( EntryPointList::Enumerator i(m_entry_points); ++i; ){
    o << "** EntryPoint: " << (*i)->name();
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEntryPoint* EntryPointMng::
findEntryPoint(const String& s)
{
  for( EntryPointList::Enumerator i(m_entry_points); ++i; )
    if ((*i)->name()==s)
      return *i;
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEntryPoint* EntryPointMng::
findEntryPoint(const String& module_name,const String& s)
{
  for( EntryPointList::Enumerator i(m_entry_points); ++i; )
    if ((*i)->name()==s && (*i)->module()->name()==module_name)
      return *i;
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

