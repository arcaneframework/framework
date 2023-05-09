// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironmentBuildInfo.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Informations pour la création d'un milieu.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/materials/MeshEnvironmentBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironmentBuildInfo::
MeshEnvironmentBuildInfo(const String& name)
: m_name(name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshEnvironmentBuildInfo::
~MeshEnvironmentBuildInfo()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironmentBuildInfo::
addMaterial(const String& name)
{
  _checkValid(name);
  m_materials.add(MatInfo(name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshEnvironmentBuildInfo::
_checkValid(const String& name)
{
  for( Integer i=0, nb_mat=m_materials.size(); i<nb_mat; ++i ){
    const String& mat_name = m_materials[i].m_name;
    if (mat_name==name)
      ARCANE_FATAL("environment named '{0}' already has a material named '{1}'",
                   m_name,name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
