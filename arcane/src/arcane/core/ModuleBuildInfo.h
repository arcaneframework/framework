// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleBuildInfo.h                                           (C) 2000-2025 */
/*                                                                           */
/* Parameters for building a module.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MODULEBUILDINFO_H
#define ARCANE_CORE_MODULEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for building a module.
 *
 * \a ModuleBuildInfo is usually used via \a BasicModule
 * (basic module) and \a AbstractModule (any module) for
 * the creation of different modules.
 * 
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT ModuleBuildInfo
{
 public:

  /*!
  * \brief Constructor from a subdomain, a mesh, and a
  * module implementation name.
  *
  * \deprecated Use the overload that takes a MeshHandle instead.
  */
  ARCANE_DEPRECATED_REASON("Y2022: use overload with meshHandle() instead of mesh")
  ModuleBuildInfo(ISubDomain* sd, IMesh* mesh, const String& name);

 public:

  //! Constructor from a subdomain, a mesh, and a module implementation name
  ModuleBuildInfo(ISubDomain* sd, const MeshHandle& mesh_handle, const String& name);

  /*!
   * \brief Constructor from a subdomain and a module implementation name
   *
   * The mesh considered is then the default mesh \a ISubDomain::defautMesh()
   */
  ModuleBuildInfo(ISubDomain* sd, const String& name);

  //! Destructor
  virtual ~ModuleBuildInfo() {}

 public:

  //! Access to the associated subdomain
  ISubDomain* subDomain() const { return m_sub_domain; }

  //! Access to the associated mesh
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

  //! Name of the implementation sought
  const String& name() const { return m_name; }

 public:

  /*!
   * \brief Access to the associated mesh.
   *
   * The mesh does not always exist if the dataset has not
   * been read yet.
   *
   * \deprecated You must use meshHandle() instead.
   */
  IMesh* mesh() const { return m_mesh_handle.mesh(); }

 private:

  //! Associated subdomain
  ISubDomain* m_sub_domain;

  //! Associated mesh
  MeshHandle m_mesh_handle;

  //! Name of the implementation sought
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
