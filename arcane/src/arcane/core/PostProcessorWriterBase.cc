// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.cc                                  (C) 2000-2014 */
/*                                                                           */
/* Classe de base d'un écrivain pour les informations de post-traitement.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/String.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/PostProcessorWriterBase.h"
#include "arcane/ISubDomain.h"
#include "arcane/ItemGroup.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
  ItemGroupCollection m_groups;
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
  m_p->m_groups = groups;
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

