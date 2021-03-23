// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironmentBuildInfo.cc                                 (C) 2000-2016 */
/*                                                                           */
/* Informations pour la création d'un milieu.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/materials/MeshEnvironmentBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
addMaterial(const String& name,const String& var_name)
{
  _checkValid(name);
  m_materials.add(MatInfo(name,var_name));
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

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
