// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlockBuildInfo.cc                                       (C) 2000-2016 */
/*                                                                           */
/* Information for creating a block.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArgumentException.h"
#include "arcane/materials/MeshBlockBuildInfo.h"
#include "arcane/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBlockBuildInfo::
MeshBlockBuildInfo(const String& name, const CellGroup& cells)
: m_name(name)
, m_cells(cells)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBlockBuildInfo::
~MeshBlockBuildInfo()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshBlockBuildInfo::
addEnvironment(IMeshEnvironment* env)
{
  if (m_environments.contains(env))
    throw ArgumentException(A_FUNCINFO,
                            String::format("environment {0} already in block {1}",
                                           env->name(), this->name()));
  m_environments.add(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
