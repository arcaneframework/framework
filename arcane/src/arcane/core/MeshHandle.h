// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshHandle.h                                                (C) 2000-2026 */
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
#include "arccore/base/AutoRef2.h"

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

  class MeshHandleRef;

 public:

  MeshHandle();
  ~MeshHandle();

  MeshHandle(const MeshHandle&);
  MeshHandle& operator=(const MeshHandle&);
  MeshHandle(MeshHandle&&) noexcept;
  MeshHandle& operator=(MeshHandle&&) noexcept;

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
  ISubDomain* subDomain() const;

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
  IUserDataList* meshUserDataList() const;

  String meshName() const;

  //! Indicates if the handle is null (it does not reference any existing mesh or not)
  bool isNull() const;

  //! Observable to be notified of destruction
  IObservable* onDestroyObservable() const;

  //! \internal
  const void* reference() const { return m_ref.get(); }

 public:

  // TODO: Make these privates and add friend on them

  //! \internal
  void _setMesh(IMesh* mesh);

  //! \internal
  void _destroyMesh();

  //! \internal
  IMesh* _internalMeshOrNull() const;

 private:

  AutoRef2<MeshHandleRef> m_ref;
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
