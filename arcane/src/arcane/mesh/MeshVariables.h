// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariables.h                                             (C) 2000-2005 */
/*                                                                           */
/* Variables containing the geometric information of a mesh.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHVARIABLES_H
#define ARCANE_MESH_MESHVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshVariable.h"
#include "arcane/VariableTypes.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Variables containing the information of
 * connectivities common to 1D, 2D, and 3D meshes.
 */
class MeshVariables
{
 public:

  MeshVariables(ISubDomain* sub_domain, const String& base_name);

  virtual ~MeshVariables() {}

 protected:

  ISubDomain* m_sub_domain;

  //! Mesh dimension
  VariableScalarInteger m_mesh_dimension;

  //! Mesh connectivity
  VariableScalarInteger m_mesh_connectivity;

  //! Names of entity families
  VariableArrayString m_item_families_name;

  //! Kind of entities in the families
  VariableArrayInteger m_item_families_kind;

  //! Name of the parent mesh
  VariableScalarString m_parent_mesh_name;

  //! Name of the parent group
  VariableScalarString m_parent_group_name;

  //! Names of parent meshes
  VariableArrayString m_child_meshes_name;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
