// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableBuildInfo.cc                                        (C) 2000-2023 */
/*                                                                           */
/* Informations pour construire une variable.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableBuildInfo.h"

#include "arcane/utils/Iostream.h"

#include "arcane/core/IModule.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/internal/IVariableMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
ISubDomain*
_getSubDomainDeprecated(const MeshHandle& handle)
{
  return handle.subDomain();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IModule* m,const String& name,int property)
: m_sub_domain(m->subDomain())
, m_module(m)
, m_mesh_handle(m->defaultMeshHandle())
, m_name(name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(ISubDomain* sd,const String& name,int property)
: m_sub_domain(sd)
, m_module(nullptr)
, m_name(name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IVariableMng* variable_mng,const String& name,int property)
: m_sub_domain(variable_mng->_internalApi()->internalSubDomain())
, m_module(nullptr)
, m_name(name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,int property)
: m_sub_domain(_getSubDomainDeprecated(mesh_handle))
, m_module(nullptr)
, m_mesh_handle(mesh_handle)
, m_name(name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IMesh* mesh,const String& name,int property)
: VariableBuildInfo(mesh->handle(),name,property)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IModule* m,const String& name,
                  const String& item_family_name,int property)
: m_sub_domain(m->subDomain())
, m_module(m)
, m_mesh_handle(m->defaultMeshHandle())
, m_name(name)
, m_item_family_name(item_family_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,
                  const String& item_family_name,int property)
: m_sub_domain(_getSubDomainDeprecated(mesh_handle))
, m_module(nullptr)
, m_mesh_handle(mesh_handle)
, m_name(name)
, m_item_family_name(item_family_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IMesh* mesh,const String& name,
                  const String& item_family_name,int property)
: VariableBuildInfo(mesh->handle(),name,item_family_name,property)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(ISubDomain* sd,const String& name,const String& mesh_name,
                  const String& item_family_name,int property)
: m_sub_domain(sd)
, m_module(nullptr)
, m_name(name)
, m_item_family_name(item_family_name)
, m_mesh_name(mesh_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IVariableMng* variable_mng,const String& name,const String& mesh_name,
                  const String& item_family_name,int property)
: m_sub_domain(variable_mng->_internalApi()->internalSubDomain())
, m_module(nullptr)
, m_name(name)
, m_item_family_name(item_family_name)
, m_mesh_name(mesh_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IItemFamily* family,const String& name,int property)
: m_sub_domain(_getSubDomainDeprecated(family->mesh()->handle()))
, m_module(nullptr)
, m_mesh_handle(family->mesh()->handle())
, m_name(name)
, m_item_family_name(family->name())
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IModule* m,const String& name,
                  const String& item_family_name,
                  const String& item_group_name,int property)
: m_sub_domain(m->subDomain())
, m_module(m)
, m_mesh_handle(m->defaultMesh()->handle())
, m_name(name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,
                  const String& item_family_name,
                  const String& item_group_name,int property)
: m_sub_domain(_getSubDomainDeprecated(mesh_handle))
, m_module(nullptr)
, m_mesh_handle(mesh_handle)
, m_name(name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IMesh* mesh,const String& name,
                  const String& item_family_name,
                  const String& item_group_name,int property)
: VariableBuildInfo(mesh->handle(),name,item_family_name,item_group_name,property)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(ISubDomain* sd,const String& name,
                  const String& mesh_name,
                  const String& item_family_name,
                  const String& item_group_name,int property)
: m_sub_domain(sd)
, m_module(nullptr)
, m_name(name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_mesh_name(mesh_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableBuildInfo::
VariableBuildInfo(IVariableMng* variable_mng,const String& name,
                  const String& mesh_name,
                  const String& item_family_name,
                  const String& item_group_name,int property)
: m_sub_domain(variable_mng->_internalApi()->internalSubDomain())
, m_module(nullptr)
, m_name(name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_mesh_name(mesh_name)
, m_property(property)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableBuildInfo::
_init()
{
  if (!m_mesh_handle.isNull()){
    m_mesh_name = m_mesh_handle.meshName();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableMng* VariableBuildInfo::
variableMng() const
{
  ARCANE_CHECK_POINTER(m_sub_domain);
  return m_sub_domain->variableMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactoryMng* VariableBuildInfo::
dataFactoryMng() const
{
  ARCANE_CHECK_POINTER(m_sub_domain);
  return m_sub_domain->application()->dataFactoryMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* VariableBuildInfo::
traceMng() const
{
  ARCANE_CHECK_POINTER(m_sub_domain);
  return m_sub_domain->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
