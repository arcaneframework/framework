// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.h                                                (C) 2000-2025 */
/*                                                                           */
/* Handle on a mesh.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHHANDLE_H
#define ARCANE_CORE_MESHHANDLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/StringView.h"
#include "arccore/base/String.h"

#include "arcane/utils/UtilsTypes.h"
#include "arcane/core/ArcaneTypes.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Handle on a mesh.
 *
 * This class uses reference counting semantics.
 *
 * This class allows managing a reference to a mesh (IMesh) before
 * it is explicitly created. This allows services and modules
 * to specify which mesh they refer to during their construction.
 *
 * It also allows associating user data with the mesh
 * via meshUserDataList().
 */
class ARCANE_CORE_EXPORT MeshHandle
{
 private:

  // Temporary: to access the constructor that uses ISubDomain.
  friend class MeshMng;

 private:

  class MeshHandleRef
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
  };

 public:

  MeshHandle();

 private:

  // TODO make accessible only to classes implementing IMeshMng.
  MeshHandle(ISubDomain* sd, const String& name);

 public:

  /*!
   * \brief Associated mesh.
   *
   * It is forbidden to call this method if the mesh has not yet been
   * created. Eventually, an exception will be raised in this case.
   *
   * If we are not sure that the mesh exists, we can test its
   * existence via hasMesh().
   *
   * \pre hasMesh() == true
   */
  IMesh* mesh() const;

  //! Indicates if the associated mesh has already been created (i.e.: mesh() is valid)
  bool hasMesh() const;

  /*!
   * \brief Returns the mesh associated with this instance.
   *
   * Unlike mesh(), this can be called even if the associated mesh has not
   * yet been created. In this case, a null pointer is returned.
   */
  IMesh* meshOrNull() const;

 public:

  //! Associated sub-domain. Null if isNull() is true.
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get ISubDomain from another way")
  ISubDomain* subDomain() const { return m_ref->subDomain(); }

 public:

  //! Associated mesh manager. nullptr if isNull() is true.
  IMeshMng* meshMng() const;

  //! Associated trace manager. nullptr if isNull() is true.
  ITraceMng* traceMng() const;

  //! Associated variable manager. nullptr if isNull() is true.
  IVariableMng* variableMng() const;

  //! Associated application. nullptr if isNull() is true.
  IApplication* application() const;

  //! Associated user data
  IUserDataList* meshUserDataList() const { return m_ref->userDataList(); }

  const String& meshName() const { return m_ref->meshName(); }

  //! Indicates if the handle is null (it does not reference any existing mesh or not)
  bool isNull() const { return m_ref->isNull(); }

  //! Observable to be notified of destruction
  IObservable* onDestroyObservable() const;

  //! \internal
  const void* reference() const { return m_ref.get(); }

 public:

  //! \internal
  void _setMesh(IMesh* mesh) { m_ref->_setMesh(mesh); }

  //! \internal
  void _destroyMesh() { m_ref->_destroyMesh(); }

  //! \internal
  IMesh* _internalMeshOrNull() const { return m_ref->mesh(); }

 private:

  Arccore::ReferenceCounter<MeshHandleRef> m_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compatibility class to hold a MeshHandle or an IMesh*.
 *
 * Eventually, the constructors and converters to IMesh* will be removed.
 */
class ARCANE_CORE_EXPORT MeshHandleOrMesh
{
 public:

  // NOTE: The constructors must not be explicit
  // to allow conversions

  //! Constructs an instance from a MeshHandle
  MeshHandleOrMesh(const MeshHandle& handle);

  /*!
   * \brief Constructs an instance from an IMesh*.
   *
   * If \a mesh is null, the associated MeshHandle will also be null.
   */
  MeshHandleOrMesh(IMesh* mesh);

  //! Associated mesh. Can be null if the mesh has not yet been created
  IMesh* mesh() const { return m_handle.meshOrNull(); }

  //! Associated mesh. Can be null if the mesh has not yet been created
  operator IMesh*() const { return mesh(); }

  //! Associated handle.
  const MeshHandle& handle() const { return m_handle; }

 private:

  MeshHandle m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
