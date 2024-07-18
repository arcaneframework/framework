// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFManager.h                                                (C) 2000-2024 */
/*                                                                           */
/* Class to handle dof families and connectivities.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DOFMANAGER_H
#define ARCANE_MESH_DOFMANAGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemConnectivityMng.h"
#include "arcane/mesh/DoFFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: mettre dans Arcane::mesh
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//TODO: renommer en DofMng
class ARCANE_MESH_EXPORT DoFManager
{
 public:

  //! Constructeur de la classe
  explicit DoFManager(IMesh* mesh);

 public:

  // TODO: a supprimer. Utiliser getFamily() à la place
  mesh::DoFFamily& family(const String& family_name, bool register_modifier_if_created = false)
  {
    return _getFamily(family_name, register_modifier_if_created);
  }

  IDoFFamily* getFamily(const String& family_name, bool register_modifier_if_created = false)
  {
    mesh::DoFFamily& dof_family = _getFamily(family_name, register_modifier_if_created);
    return &dof_family;
  }

  IItemConnectivityMng* connectivityMng() const { return m_connectivity_mng; }

 private:

  Arcane::IMesh* m_mesh;
  IItemConnectivityMng* m_connectivity_mng;

 private:

  mesh::DoFFamily& _getFamily(const String& family_name, bool register_modifier_if_created = false)
  {
    bool create_if_needed = true;
    IItemFamily* item_family = m_mesh->findItemFamily(Arcane::IK_DoF, family_name, create_if_needed, register_modifier_if_created);
    mesh::DoFFamily* dof_family = static_cast<mesh::DoFFamily*>(item_family);
    return *dof_family;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* DOFMANAGER_H_ */
