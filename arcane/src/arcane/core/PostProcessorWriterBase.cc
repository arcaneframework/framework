// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.cc                                  (C) 2000-2026 */
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

PostProcessorWriterCommonBase::
PostProcessorWriterCommonBase()
: m_p(new PostProcessorWriterBasePrivate())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PostProcessorWriterCommonBase::
~PostProcessorWriterCommonBase()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PostProcessorWriterBase::
PostProcessorWriterBase(const ServiceBuildInfo& sbi)
: BasicService(sbi)
{
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterCommonBase::
setBaseDirectoryName(const String& dirname)
{
  m_p->m_base_dirname = dirname;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& PostProcessorWriterCommonBase::
baseDirectoryName()
{
  return m_p->m_base_dirname;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterCommonBase::
setBaseFileName(const String& filename)
{
  m_p->m_base_filename = filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& PostProcessorWriterCommonBase::
baseFileName()
{
  return m_p->m_base_filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterCommonBase::
setTimes(RealConstArrayView times)
{
  m_p->m_times = times;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterCommonBase::
setVariables(VariableCollection variables)
{
  m_p->m_variables = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PostProcessorWriterCommonBase::
setGroups(ItemGroupCollection groups)
{
  m_p->m_groups.clone(groups);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Real> PostProcessorWriterCommonBase::
times()
{
  return m_p->m_times;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableCollection PostProcessorWriterCommonBase::
variables()
{
  return m_p->m_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupCollection PostProcessorWriterCommonBase::
groups()
{
  return m_p->m_groups;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String IPostProcessorWriter::
getBaseDirectoryName()
{
  return baseDirectoryName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String IPostProcessorWriter::
getBaseFileName()
{
  return baseFileName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
