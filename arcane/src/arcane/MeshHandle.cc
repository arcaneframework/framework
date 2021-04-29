// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.cc                                               (C) 2000-2020 */
/*                                                                           */
/* Handle sur un maillage.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshHandle.h"

#include "arcane/utils/UserDataList.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO rendre ce constructeur privé à IMeshMng pour éviter d'avoir des
// doublons possibles pour les MeshHandle associés au même nom de maillage.
MeshHandle::MeshHandleRef::
MeshHandleRef(ISubDomain* sd,const String& name)
: m_mesh_name(name)
, m_sub_domain(sd)
, m_is_null(name.null())
{
  if (!m_is_null)
    m_user_data_list = new UserDataList();
  ARCANE_CHECK_POINTER(sd);
  m_trace_mng = sd->traceMng();
  m_mesh_mng = sd->meshMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::MeshHandleRef::
~MeshHandleRef()
{
  delete m_user_data_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshHandle::MeshHandleRef::
_setMesh(IMesh* mesh)
{
  m_mesh_ptr = mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshHandle::MeshHandleRef::
_setMeshBase(IMeshBase* mesh_base)
{
  m_mesh_base_ptr = mesh_base;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshHandle::MeshHandleRef::
_destroyMesh()
{
  // TODO: protéger les appels multiples
  IMesh* mesh = m_mesh_ptr;
  if (!mesh)
    return;
  m_user_data_list->clear();
  // Attention à ne mettre à nul que à la fin de cette routine car les
  // utilisateurs de \a m_user_data peuvent avoir besoin de ce MeshHandle.
  m_mesh_ptr = nullptr;
  delete mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::
MeshHandle(ISubDomain* sd,const String& name)
: m_ref(new MeshHandleRef(sd,name))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::
MeshHandle()
: m_ref(new MeshHandleRef())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Gestionnaire de maillage associé. Null si isNull() est vrai.
IMeshMng* MeshHandle::
meshMng() const
{
  return m_ref->meshMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshHandle::
hasMesh() const
{
  IMesh* m = m_ref->mesh();
  return m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshHandle::
mesh() const
{
  IMesh* m = m_ref->mesh();
  if (m)
    return m;
  // A terme, faire un fatal si le maillage est nul. Pour des raisons de
  // compatibilité avec l'existant, on retourne 'nullptr'.
  bool do_fatal = false;
  if (do_fatal)
    ARCANE_FATAL("Invalid call for null mesh. Call MeshHandle::hasMesh() before to make sure mesh is valid");
  return nullptr;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshBase* MeshHandle::
meshBase() const
{
  auto imesh = mesh();
  if (imesh) {
    return static_cast<IMeshBase*>(imesh);
  }
  else return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* MeshHandle::
traceMng() const
{
  return m_ref->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

