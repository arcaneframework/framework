// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TemporaryVariableBuildInfo.cc                               (C) 2000-2020 */
/*                                                                           */
/* Informations pour construire une variable temporaire.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/TemporaryVariableBuildInfo.h"
#include "arcane/IModule.h"
#include "arcane/IVariable.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TemporaryVariableBuildInfo::
TemporaryVariableBuildInfo(IModule* m,const String& name)
: VariableBuildInfo(m,_generateName(m->subDomain()->variableMng(),name),property())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TemporaryVariableBuildInfo::
TemporaryVariableBuildInfo(ISubDomain* sd,const String& name)
: VariableBuildInfo(sd,_generateName(sd->variableMng(),name),property())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TemporaryVariableBuildInfo::
TemporaryVariableBuildInfo(IModule* m,const String& name,const String& item_family_name)
: VariableBuildInfo(m,_generateName(m->subDomain()->variableMng(),name),item_family_name,property())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TemporaryVariableBuildInfo::
TemporaryVariableBuildInfo(IMesh* mesh,const String& name)
: VariableBuildInfo(mesh,_generateName(mesh->variableMng(),name),property())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TemporaryVariableBuildInfo::
TemporaryVariableBuildInfo(IMesh* mesh,const String& name,const String& item_family_name)
: VariableBuildInfo(mesh,_generateName(mesh->variableMng(),name),item_family_name,property())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int TemporaryVariableBuildInfo::
property()
{
  return IVariable::PTemporary | IVariable::PNoDump | IVariable::PNoRestore;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String TemporaryVariableBuildInfo::
_generateName(IVariableMng* vm,const String& name)
{
  ARCANE_UNUSED(vm);
  return name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

