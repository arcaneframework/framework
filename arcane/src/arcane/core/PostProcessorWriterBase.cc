// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un écrivain pour les informations de post-traitement.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/PostProcessorWriterBase.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PostProcessorWriterBasePrivate
{
 public:

  PostProcessorWriterBasePrivate();

 public:

  String m_base_dirname;
  String m_base_filename;
  VariableCollection m_variables;
  ItemGroupList m_groups;
  SharedArray<Real> m_times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PostProcessorWriterBasePrivate::
PostProcessorWriterBasePrivate()
: m_base_dirname(".")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PostProcessorWriterBase::
PostProcessorWriterBase(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_p(new PostProcessorWriterBasePrivate())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PostProcessorWriterBase::
~PostProcessorWriterBase()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterBase::
setBaseDirectoryName(const String& dirname)
{
  m_p->m_base_dirname = dirname;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& PostProcessorWriterBase::
baseDirectoryName()
{
  return m_p->m_base_dirname;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterBase::
setBaseFileName(const String& filename)
{
  m_p->m_base_filename = filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& PostProcessorWriterBase::
baseFileName()
{
  return m_p->m_base_filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterBase::
setTimes(RealConstArrayView times)
{
  m_p->m_times = times;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterBase::
setVariables(VariableCollection variables)
{
  m_p->m_variables = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterBase::
setGroups(ItemGroupCollection groups)
{
  m_p->m_groups.clone(groups);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealConstArrayView PostProcessorWriterBase::
times()
{
  return m_p->m_times;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableCollection PostProcessorWriterBase::
variables()
{
  return m_p->m_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupCollection PostProcessorWriterBase::
groups()
{
  return m_p->m_groups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
