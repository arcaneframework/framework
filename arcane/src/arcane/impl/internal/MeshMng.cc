// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMng.cc                                                  (C) 2000-2023 */
/*                                                                           */
/* Classe gérant la liste des maillages.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/MeshMng.h"

#include "arcane/utils/String.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/impl/internal/MeshFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMng::
MeshMng(IApplication* app,IVariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
, m_mesh_factory_mng(new MeshFactoryMng(app,this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMng::
~MeshMng()
{
  delete m_mesh_factory_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMng::
addMesh(IMesh* mesh)
{
  ARCANE_CHECK_POINTER(mesh);
  String name = mesh->name();
  info() << "Add mesh name=" << mesh->name() << " count="  << meshes().size();
  //std::cout << " ** ** ADD_MESH name=" << name << " this=" << mesh << '\n';
  // Regarde si un handle sur le maillage est déjà présent.
  // Si c'est le cas, l'utilise.
  MeshHandle* handle_ptr = findMeshHandle(name,false);
  if (handle_ptr){
    // Vérifie que le handle n'a pas déjà de maillage associé ou si c'est le
    // cas que c'est le même maillage que celui qu'on a déjà
    // TODO: cela ne devrait pas être autorisé quand même car le maillage
    // va être ajouté plusieurs fois dans la liste des maillages via
    // m_meshes.add().
    if (handle_ptr->hasMesh()){
      IMesh* current_mesh = handle_ptr->mesh();
      //std::cout << " ** ** FOUND_HANDLE ref=" << handle_ptr->reference()
      //          << " old_mesh=" << current_mesh << " new_mesh=" << mesh << '\n';
      if (current_mesh!=mesh)
        ARCANE_FATAL("MeshHandle '{0}' already have mesh",name);
    }
    else
      handle_ptr->_setMesh(mesh);
  }
  else{
    MeshHandle handle = _addMeshHandle(name);
    handle._setMesh(mesh);
  }
  _rebuildMeshList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshMng::
getMesh(Integer index) const
{
  return m_meshes_handle[index].mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshMng::
getPrimaryMesh(Integer index) const
{
  return getMesh(index)->toPrimaryMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle* MeshMng::
findMeshHandle(const String& name,bool throw_exception)
{
  for( auto& handle : m_meshes_handle )
    if (handle.meshName()==name)
      return &handle;
  if (throw_exception)
    ARCANE_FATAL("no MeshHandle named '{0}' found",name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle MeshMng::
findMeshHandle(const String& name)
{
  MeshHandle* handle = findMeshHandle(name,true);
  //std::cout << " ** ** FOUND_HANDLE name=" << name << " ptr=" << handle->reference()
  //          << " mesh=" << handle->mesh() << '\n';
  return MeshHandle(*handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle MeshMng::
createMeshHandle(const String& name)
{
  MeshHandle* old_handle = findMeshHandle(name,false);
  if (old_handle)
    ARCANE_FATAL("mesh handle already exists for name '{0}'",name);
  return _addMeshHandle(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshMng::
findMesh(const String& name,bool throw_exception)
{
  const MeshHandle* handle = findMeshHandle(name,throw_exception);
  if (!handle)
    return nullptr;
  IMesh* mesh = handle->mesh();
  if (!mesh && throw_exception)
    ARCANE_FATAL("no mesh named '{0}' found",name);
  return mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<IMesh*> MeshMng::
meshes() const
{
  return m_meshes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMng::
destroyMeshes()
{
  for( MeshHandle& handle : m_meshes_handle ){
    IMesh* x = handle.mesh();
    if (x){
      m_variable_mng->_internalApi()->detachMeshVariables(x);
      handle._destroyMesh();
    }
  }
  m_meshes_handle.clear();
  m_meshes.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMng::
destroyMesh(MeshHandle handle)
{
  if (!handle.hasMesh())
    return;
  IMesh* mesh = handle.mesh();
  if (!mesh->isPrimaryMesh())
    ARCANE_FATAL("Can not destroy mesh '{0}' because it is not a primary mesh",mesh->name());
  _destroyMesh(mesh->toPrimaryMesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMng::
_destroyMesh(IPrimaryMesh* primary_mesh)
{
  IMesh* m = primary_mesh;
  String name = m->name();
  MeshHandle handle = m->handle();
  m_variable_mng->_internalApi()->detachMeshVariables(m);
  handle._destroyMesh();

  // Supprime le maillage de la liste.
  for( Integer i=0, n=m_meshes.size(); i<n; ++i ){
    if (m_meshes[i]==m){
      m_meshes.remove(i);
      break;
    }
  }

  // Supprime le MeshHandle de la liste
  for( Integer i=0, n=m_meshes_handle.size(); i<n; ++i ){
    if (m_meshes_handle[i].meshName()==name){
      m_meshes_handle.remove(i);
      break;
    }
  }

  _rebuildMeshList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle MeshMng::
_addMeshHandle(const String& name)
{
  //std::cout << "_ADD_MESH_HANDLE handle=" << name << "\n";
  MeshHandle handle(m_variable_mng->_internalApi()->internalSubDomain(),name);
  m_meshes_handle.add(handle);
  return handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMng::
_rebuildMeshList()
{
  m_meshes.clear();
  for( MeshHandle& h : m_meshes_handle ){
    if (!h.hasMesh())
      continue;
    m_meshes.add(h.mesh());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshFactoryMng* MeshMng::
meshFactoryMng() const
{
  return m_mesh_factory_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle MeshMng::
createDefaultMeshHandle(const String& name)
{
  m_default_mesh_handle = createMeshHandle(name);
  return m_default_mesh_handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
