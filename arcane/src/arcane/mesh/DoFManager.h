// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFManager.h                                                (C) 2000-2015 */
/*                                                                           */
/* Comment on file content.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOFMANAGER_H
#define ARCANE_DOFMANAGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/IMesh.h"
#include "arcane/mesh/DoFFamily.h"

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//TODO: renommer en DofMng
class DoFManager
{
 public:

  /** Constructeur de la classe */
  DoFManager(IMesh* mesh, IItemConnectivityMng* connectivity_mng)
  : m_mesh(mesh) , m_connectivity_mng(connectivity_mng){}

  /** Destructeur de la classe */
  virtual ~DoFManager() {}

 public:

  mesh::DoFFamily & family(const Arcane::String& family_name)
  {
    bool create_if_needed = true;
    Arcane::IItemFamily* item_family = m_mesh->findItemFamily(Arcane::IK_DoF,family_name,create_if_needed);
    mesh::DoFFamily* dof_family = static_cast<mesh::DoFFamily*>(item_family);
    return *dof_family;
  }
  IItemConnectivityMng* connectivityMng() const { return m_connectivity_mng; }

 private:

  Arcane::IMesh* m_mesh;
  IItemConnectivityMng* m_connectivity_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* DOFMANAGER_H_ */
