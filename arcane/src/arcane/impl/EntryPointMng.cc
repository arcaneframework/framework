// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EntryPointMng.cc                                            (C) 2000-2023 */
/*                                                                           */
/* Entry Point Manager.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IEntryPointMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Entry Point Manager.
 */
class EntryPointMng
: public TraceAccessor
, public IEntryPointMng
{
 public:

  explicit EntryPointMng(ISubDomain*);
  ~EntryPointMng() override;

 public:

  void addEntryPoint(IEntryPoint*) override;
  void dumpList(std::ostream&) override;
  IEntryPoint* findEntryPoint(const String& s) override;
  IEntryPoint* findEntryPoint(const String& module_name, const String& s) override;
  EntryPointCollection entryPoints() override { return m_entry_points; }

 private:

  EntryPointList m_entry_points; //!< List of entry points
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
  for (EntryPointList::Enumerator i(m_entry_points); ++i;)
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
dumpList(std::ostream& o)
{
  o << "** EntryPointMng::dump_list: " << m_entry_points.count();
  o << '\n';
  for (EntryPointList::Enumerator i(m_entry_points); ++i;) {
    o << "** EntryPoint: " << (*i)->name();
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEntryPoint* EntryPointMng::
findEntryPoint(const String& s)
{
  for (EntryPointList::Enumerator i(m_entry_points); ++i;)
    if ((*i)->name() == s)
      return *i;
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEntryPoint* EntryPointMng::
findEntryPoint(const String& module_name, const String& s)
{
  for (EntryPointList::Enumerator i(m_entry_points); ++i;)
    if ((*i)->name() == s && (*i)->module()->name() == module_name)
      return *i;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
