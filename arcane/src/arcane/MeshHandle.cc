// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.cc                                               (C) 2000-2022 */
/*                                                                           */
/* Handle sur un maillage.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshHandle.h"

#include "arcane/utils/UserDataList.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Observable.h"
#include "arcane/utils/ValueConvert.h"

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
  m_variable_mng = sd->variableMng();
  m_on_destroy_observable = new Observable();
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DO_FATAL_IN_MESHHANDLE",true))
    m_do_fatal_in_mesh_method = v.value()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::MeshHandleRef::
~MeshHandleRef()
{
  delete m_user_data_list;
  delete m_on_destroy_observable;
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
_destroyMesh()
{
  // TODO: protéger les appels multiples
  IMesh* mesh = m_mesh_ptr;
  if (!mesh)
    return;
  m_on_destroy_observable->notifyAllObservers();
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
  bool do_fatal = true; //m_ref->isDoFatalInMeshMethod();
  if (do_fatal)
    ARCANE_FATAL("Invalid call for null mesh. Call MeshHandle::hasMesh() before to make sure mesh is valid");
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshHandle::
meshOrNull() const
{
  return m_ref->mesh();
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

IVariableMng* MeshHandle::
variableMng() const
{
  return m_ref->variableMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication* MeshHandle::
application() const
{
  return m_ref->subDomain()->application();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* MeshHandle::
onDestroyObservable() const
{
  return m_ref->onDestroyObservable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

