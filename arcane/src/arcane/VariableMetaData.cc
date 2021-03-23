// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMetaData.cc                                         (C) 2000-2018 */
/*                                                                           */
/* Meta-données sur une variable.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"

#include "arcane/VariableMetaData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableMetaData::
VariableMetaData(const String& base_name,const String& mesh_name,
                 const String& item_family_name,const String& item_group_name,
                 bool is_partial)
: m_base_name(base_name)
, m_mesh_name(mesh_name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_is_partial(is_partial)
{
  _buildFullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMetaData::
_buildFullName()
{
  StringBuilder full_name_b;
  if (!m_mesh_name.null()){
    full_name_b += m_mesh_name;
    full_name_b += "_";
  }
  if (!m_item_family_name.null()){
    full_name_b += m_item_family_name;
    full_name_b += "_";
  }
  full_name_b += m_base_name;
  m_full_name = full_name_b.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

