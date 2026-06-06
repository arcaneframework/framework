// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMng.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Mesh manager interface.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHMNG_H
#define ARCANE_CORE_IMESHMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh manager interface.
 *
 * This interface manages a list of meshes and allows creating meshes
 * or retrieving an existing mesh by its name.
 *
 * Mesh creation is done via 'IMeshFactoryMng' whose instance
 * can be retrieved via meshFactoryMng(). Effective mesh creation cannot
 * take place until after reading the dataset. However, it is possible
 * to create a reference (via createMeshHandle()) to a mesh at any
 * time.
 */
class ARCANE_CORE_EXPORT IMeshMng
{
 public:

  //! Frees the resources.
  virtual ~IMeshMng() = default;

 public:

  //! Trace manager associated with this manager
  virtual ITraceMng* traceMng() const = 0;

  //! Mesh factory associated with this manager
  virtual IMeshFactoryMng* meshFactoryMng() const = 0;

  //! Variable manager associated with this manager
  virtual IVariableMng* variableMng() const = 0;

 public:

  /*!
   * \brief Searches for the mesh with name \a name.
   *
   * If the mesh is not found, the method throws an exception
   * if \a throw_exception is \a true or returns *nullptr* if \a throw_exception
   * is \a false.
   */
  virtual MeshHandle* findMeshHandle(const String& name, bool throw_exception) = 0;

  /*!
   * \brief Searches for the mesh with name \a name.
   *
   * If the mesh is not found, the method throws an exception.
   */
  virtual MeshHandle findMeshHandle(const String& name) = 0;

  /*!
   * \brief Creates and returns a handle for a mesh with name \a name.
   *
   * Throws an exception if a handle associated with this name already exists.
   */
  virtual MeshHandle createMeshHandle(const String& name) = 0;

  /*!
   * \brief Destroys the mesh associated with \a handle.
   *
   * The mesh must be a mesh implementing IPrimaryMesh.
   *
   * \warning \a handle must no longer be used after this call
   * and the associated mesh either. If references to these two
   * objects remain, the behavior is undefined.
   */
  virtual void destroyMesh(MeshHandle handle) = 0;

  //! Handle for the default mesh.
  virtual MeshHandle defaultMeshHandle() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
