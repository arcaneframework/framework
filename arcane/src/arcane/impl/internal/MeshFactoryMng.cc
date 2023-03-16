// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshFactoryMng.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de fabriques de maillages.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/MeshFactoryMng.h"

#include "arcane/impl/internal/MeshMng.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/MeshHandle.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IMeshFactory.h"
#include "arcane/ItemGroup.h"
#include "arcane/MeshBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{

Ref<IMeshFactory>
_getMeshFactory(IApplication* app,const MeshBuildInfo& mbi)
{
  String factory_name = mbi.factoryName();
  ServiceBuilder<IMeshFactory> service_builder(app);
  Ref<IMeshFactory> mf = service_builder.createReference(factory_name,SB_AllowNull);
  if (!mf){
    StringUniqueArray valid_names;
    service_builder.getServicesNames(valid_names);
    String valid_str = String::join(", ",valid_names);
    ARCANE_FATAL("No mesh factory named '{0}' found for creating mesh '{1}'."
                 " Valid values are {2}",
                 factory_name,mbi.name(),valid_str);
  }
  return mf;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshFactoryMng::
MeshFactoryMng(IApplication* app,MeshMng* mesh_mng)
: m_application(app)
, m_mesh_mng(mesh_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMng* MeshFactoryMng::
meshMng() const
{
  return m_mesh_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshFactoryMng::
_checkValidBuildInfo(const MeshBuildInfo& build_info)
{
  if (build_info.name().empty())
    ARCANE_FATAL("Can not create mesh with null name()");
  if (build_info.factoryName().empty())
    ARCANE_FATAL("Can not create mesh with null factoryName()");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshFactoryMng::
createMesh(const MeshBuildInfo& build_info)
{
  _checkValidBuildInfo(build_info);
  ItemGroup parent_group = build_info.parentGroup();
  if (parent_group.null())
    return _createMesh(build_info);
  return _createSubMesh(build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshFactoryMng::
_createMesh(const MeshBuildInfo& build_info)
{
  if (build_info.parallelMngRef().isNull())
    ARCANE_FATAL("Can not create mesh with null parallelMngRef()");
  String name = build_info.name();
  IMeshMng* mesh_mng = m_mesh_mng;
  MeshHandle* old_mesh_handle = mesh_mng->findMeshHandle(name,false);
  if (old_mesh_handle){
    IMesh* old_mesh = old_mesh_handle->_internalMeshOrNull();
    if (old_mesh){
      IPrimaryMesh* prm = dynamic_cast<IPrimaryMesh*>(old_mesh);
      if (!prm)
        ARCANE_FATAL("A mesh with same name already exists and is not a primary mesh");
      return prm;
    }
  }

  Ref<IMeshFactory> mesh_factory(_getMeshFactory(m_application,build_info));

  // Enregistre d'abord le handle car createMesh() l'utilise.
  // Il est possible qu'un handle sur ce maillage soit déjà créé
  // (C'est notamment le cas pour le maillage par défaut).
  // Dans ce cas, vérifie que le maillage à créer n'a pas de handle
  // et si findMeshHandle() retourne null, le handle sera créé.
  MeshHandle* mh = mesh_mng->findMeshHandle(name,false);
  if (!mh)
    mesh_mng->createMeshHandle(name);
  IPrimaryMesh* mesh = mesh_factory->createMesh(m_mesh_mng,build_info);
  m_mesh_mng->addMesh(mesh);
  mesh->build();
  return mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshFactoryMng::
_createSubMesh(const MeshBuildInfo& orig_build_info)
{
  ItemGroup group = orig_build_info.parentGroup();
  IMesh* mesh = group.mesh();
  ARCANE_CHECK_POINTER(mesh);

  if (mesh->meshMng() != m_mesh_mng)
    ARCANE_FATAL("This mesh has not been created by this factory");
  if (group.isAllItems())
    ARCANE_FATAL("Cannot create sub-mesh from all-items group");

  MeshBuildInfo build_info(orig_build_info);
  String name = build_info.name();
  // Pour un sous-maillage, le IParallelMng est obligatoirement celui
  // du maillage parent
  build_info.addParallelMng(makeRef(mesh->parallelMng()));

  Ref<IMeshFactory> mesh_factory(_getMeshFactory(m_application,build_info));

  // Enregistre d'abord le handle car createMesh() l'utilise.
  // TODO: faire cela dans le constructeur de 'DynamicMesh' et positionner
  // le handle aussi dans le constructeur.
  m_mesh_mng->createMeshHandle(name);
  IPrimaryMesh* sub_mesh = mesh_factory->createMesh(m_mesh_mng,build_info);
  m_mesh_mng->addMesh(sub_mesh);

  sub_mesh->defineParentForBuild(mesh,group);
  sub_mesh->build();
  mesh->addChildMesh(sub_mesh);
  return sub_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
