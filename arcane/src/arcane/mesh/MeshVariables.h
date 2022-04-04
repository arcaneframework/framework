// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariables.h                                             (C) 2000-2005 */
/*                                                                           */
/* Variables pour contenant les informations géométrique d'un maillage.      */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Variables contenant les informations de
 * connectivités communes aux maillages 1D, 2D et 3D.
 */
class MeshVariables
{
 public:

  MeshVariables(ISubDomain* sub_domain,const String& base_name);

  virtual ~MeshVariables() {}

 protected:
  
  ISubDomain* m_sub_domain;

  //! Dimension du maillage
  VariableScalarInteger m_mesh_dimension;

  //! Connectivité du maillage
  VariableScalarInteger m_mesh_connectivity;

  //! Noms des familles d'entités
  VariableArrayString m_item_families_name;

  //! Genre des entités des familles
  VariableArrayInteger m_item_families_kind;

  //! Nom du maillage parent
  VariableScalarString m_parent_mesh_name;

  //! Nom du groupe parent
  VariableScalarString m_parent_group_name;

  //! Noms des maillages parentés
  VariableArrayString m_child_meshes_name;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
