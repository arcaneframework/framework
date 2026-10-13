// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Handle on a mesh.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshHandle.h"

#include "arcane/utils/UserDataList.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Observable.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshHandle::MeshHandleRef
: public Arccore::ReferenceCounterImpl
{
 public:

  MeshHandleRef()
  : m_is_null(true)
  {}
  MeshHandleRef(ISubDomain* sd, const String& name);
  ~MeshHandleRef();

 public:

  const String& meshName() const { return m_mesh_name; }
  bool isNull() const { return m_is_null; }
  IMesh* mesh() const { return m_mesh_ptr; }
  IMeshBase* meshBase() const { return m_mesh_base_ptr; }
  ISubDomain* subDomain() const { return m_sub_domain; }
  IMeshMng* meshMng() const { return m_mesh_mng; }
  ITraceMng* traceMng() const { return m_trace_mng; }
  IVariableMng* variableMng() const { return m_variable_mng; }
  IUserDataList* userDataList() const { return m_user_data_list; }
  Observable* onDestroyObservable() const { return m_on_destroy_observable; }
  bool isDoFatalInMeshMethod() const { return m_do_fatal_in_mesh_method; }

 public:

  void _destroyMesh();
  void _setMesh(IMesh* mesh);

 public:

  void addRef()
  {
    ++m_nb_ref;
  }
  void removeRef()
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      delete this;
  }

 private:

  String m_mesh_name;
  IMesh* m_mesh_ptr = nullptr;
  IMeshBase* m_mesh_base_ptr = nullptr;
  // For now we need it, but it should be removed
  ISubDomain* m_sub_domain = nullptr;
  IUserDataList* m_user_data_list = nullptr;
  IMeshMng* m_mesh_mng = nullptr;
  ITraceMng* m_trace_mng = nullptr;
  IVariableMng* m_variable_mng = nullptr;
  Observable* m_on_destroy_observable = nullptr;
  bool m_is_null = true;
  bool m_do_fatal_in_mesh_method = false;
  std::atomic<Int32> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO make this constructor private to IMeshMng to avoid
// possible duplicates for MeshHandles associated with the same mesh name.
MeshHandle::MeshHandleRef::
MeshHandleRef(ISubDomain* sd, const String& name)
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
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DO_FATAL_IN_MESHHANDLE", true))
    m_do_fatal_in_mesh_method = v.value() != 0;
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
  // TODO: protect against multiple calls
  IMesh* mesh = m_mesh_ptr;
  if (!mesh)
    return;
  m_on_destroy_observable->notifyAllObservers();
  m_user_data_list->clear();
  // Be careful not to set to null until the end of this routine because the
  // users of \a m_user_data might need this MeshHandle.
  m_mesh_ptr = nullptr;
  delete mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::
MeshHandle(ISubDomain* sd, const String& name)
: m_ref(new MeshHandleRef(sd, name))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle::
MeshHandle()
: m_ref(new MeshHandleRef())
{
}

MeshHandle::
MeshHandle(const MeshHandle& x)
: m_ref(x.m_ref)
{
}

MeshHandle::
MeshHandle(MeshHandle&& x) noexcept
: m_ref(std::move(x.m_ref))
{
}

MeshHandle& MeshHandle::
operator=(const MeshHandle& x)
{
  if (&x != this)
    m_ref = x.m_ref;
  return (*this);
}

MeshHandle& MeshHandle::
operator=(MeshHandle&& x) noexcept
{
  m_ref = std::move(x.m_ref);
  return (*this);
}

MeshHandle::
~MeshHandle()
{
  // The destructor needs to be here because MeshHandleRef is not known
  // in the class definition.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  // Eventually, make a fatal error if the mesh is null. For reasons of
  // compatibility with the existing code, we return 'nullptr'.
  bool do_fatal = m_ref->isDoFatalInMeshMethod();
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

IUserDataList* MeshHandle::meshUserDataList() const
{
  return m_ref->userDataList();
}

String MeshHandle::meshName() const
{
  return m_ref->meshName();
}

//! Indicates if the handle is null (it does not reference any existing mesh or not)
bool MeshHandle::isNull() const
{
  return m_ref->isNull();
}

//! \internal
void MeshHandle::
_setMesh(IMesh* mesh)
{
  m_ref->_setMesh(mesh);
}

//! \internal
void MeshHandle::
_destroyMesh()
{
  m_ref->_destroyMesh();
}

//! \internal
IMesh* MeshHandle::
_internalMeshOrNull() const
{
  return m_ref->mesh();
}

ISubDomain* MeshHandle::
subDomain() const
{
  return m_ref->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandleOrMesh::
MeshHandleOrMesh(const MeshHandle& handle)
: m_handle(handle)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandleOrMesh::
MeshHandleOrMesh(IMesh* mesh)
{
  if (mesh)
    m_handle = mesh->handle();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
