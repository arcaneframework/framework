// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMng.h                                                   (C) 2000-2020 */
/*                                                                           */
/* Classe gérant la liste des maillages.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_MESHMNG_H
#define ARCANE_IMPL_INTERNAL_MESHMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/IMeshMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;
class IVariableMng;
class IPrimaryMesh;
class MeshFactoryMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT MeshMng
: public TraceAccessor
, public IMeshMng
{
 public:

  MeshMng(IApplication* app,IVariableMng* vm);
  ~MeshMng();

 public:

  MeshMng(const MeshMng& rhs) = delete;
  MeshMng& operator=(const MeshMng& rhs) = delete;

 public:

  ITraceMng* traceMng() const override { return TraceAccessor::traceMng(); }
  IMeshFactoryMng* meshFactoryMng() const override;
  IVariableMng* variableMng() const override { return m_variable_mng; }

  MeshHandle* findMeshHandle(const String& name,bool throw_exception) override;
  MeshHandle findMeshHandle(const String& name) override;
  MeshHandle createMeshHandle(const String& name) override;
  void destroyMesh(MeshHandle handle) override;

 public:

  void addMesh(IMesh* mesh);
  void destroyMeshes();
  ConstArrayView<IMesh*> meshes() const;
  //TODO: a supprimer
  IMesh* findMesh(const String& name,bool throw_exception);
  //TODO: a supprimer
  IMesh* getMesh(Integer index) const;
  //TODO: a supprimer
  IPrimaryMesh* getPrimaryMesh(Integer index) const;

 protected:

  MeshHandle _addMeshHandle(const String& name);
  void _destroyMesh(IPrimaryMesh* primary_mesh);
  void _rebuildMeshList();

 private:

  UniqueArray<IMesh*> m_meshes;
  UniqueArray<MeshHandle> m_meshes_handle;
  IVariableMng* m_variable_mng;
  MeshFactoryMng* m_mesh_factory_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
