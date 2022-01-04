// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGenerationInfo.cc                              (C) 2000-2021 */
/*                                                                           */
/* Informations sur la génération des maillages cartésiens.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/CartesianMeshGenerationInfo.h"

#include "arcane/utils/String.h"
#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/AutoDestroyUserData.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IMesh.h"
#include "arcane/Properties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshGenerationInfo::
CartesianMeshGenerationInfo(IMesh* mesh)
: m_mesh(mesh)
, m_mesh_dimension(mesh->dimension())
{
  m_global_nb_cells = Int64ArrayView(NB_DIM, m_global_nb_cell_ptr);
  m_sub_domain_offsets = Int32ArrayView(NB_DIM, m_sub_domain_offset_ptr);
  m_nb_sub_domains = Int32ArrayView(NB_DIM, m_nb_sub_domain_ptr);
  m_own_nb_cells = Int32ArrayView(NB_DIM, m_own_nb_cell_ptr);
  m_own_cell_offsets = Int64ArrayView(NB_DIM, m_own_cell_offset_ptr);

  m_global_nb_cells.fill(-1);
  m_sub_domain_offsets.fill(-1);
  m_nb_sub_domains.fill(-1);
  m_own_nb_cells.fill(-1);
  m_own_cell_offsets.fill(-1);

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setOwnCellOffsets(Int64 x,Int64 y,Int64 z)
{
  m_own_cell_offsets[0] = x;
  m_own_cell_offsets[1] = y;
  m_own_cell_offsets[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt64("OwnCellOffsetX", x);
  p->setInt64("OwnCellOffsetY", y);
  p->setInt64("OwnCellOffsetZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setGlobalNbCells(Int64 x, Int64 y, Int64 z)
{
  m_global_nb_cells[0] = x;
  m_global_nb_cells[1] = y;
  m_global_nb_cells[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt64("GlobalNbCellX", x);
  p->setInt64("GlobalNbCellY", y);
  p->setInt64("GlobalNbCellZ", z);

  m_global_nb_cell = x;
  if (y > (-1))
    m_global_nb_cell *= y;
  if (z > (-1))
    m_global_nb_cell *= z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setSubDomainOffsets(Int32 x, Int32 y, Int32 z)
{
  m_sub_domain_offsets[0] = x;
  m_sub_domain_offsets[1] = y;
  m_sub_domain_offsets[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt32("SubDomainOffsetX", x);
  p->setInt32("SubDomainOffsetY", y);
  p->setInt32("SubDomainOffsetZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setNbSubDomains(Int32 x, Int32 y, Int32 z)
{
  m_nb_sub_domains[0] = x;
  m_nb_sub_domains[1] = y;
  m_nb_sub_domains[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt32("NbSubDomainX", x);
  p->setInt32("NbSubDomainY", y);
  p->setInt32("NbSubDomainZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setOwnNbCells(Int32 x, Int32 y, Int32 z)
{
  m_own_nb_cells[0] = x;
  m_own_nb_cells[1] = y;
  m_own_nb_cells[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt32("OwnNbCellX", x);
  p->setInt32("OwnNbCellY", y);
  p->setInt32("OwnNbCellZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setFirstOwnCellUniqueId(Int64 uid)
{
  m_first_own_cell_unique_id = uid;

  Properties* p = m_mesh->properties();
  p->setInt64("CartesianFirstOnwCellUniqueId", uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
_init()
{
  Properties* p = m_mesh->properties();
  m_global_nb_cells[MD_DirX] = p->getInt64WithDefault("GlobalNbCellX", -1);
  m_global_nb_cells[MD_DirY] = p->getInt64WithDefault("GlobalNbCellY", -1);
  m_global_nb_cells[MD_DirZ] = p->getInt64WithDefault("GlobalNbCellZ", -1);

  m_own_nb_cells[MD_DirX] = p->getInt32WithDefault("OwnNbCellX", -1);
  m_own_nb_cells[MD_DirY] = p->getInt32WithDefault("OwnNbCellY", -1);
  m_own_nb_cells[MD_DirZ] = p->getInt32WithDefault("OwnNbCellZ", -1);

  m_sub_domain_offsets[MD_DirX] = p->getInt32WithDefault("SubDomainOffsetX", -1);
  m_sub_domain_offsets[MD_DirY] = p->getInt32WithDefault("SubDomainOffsetY", -1);
  m_sub_domain_offsets[MD_DirZ] = p->getInt32WithDefault("SubDomainOffsetZ", -1);

  m_own_cell_offsets[MD_DirX] = p->getInt64WithDefault("OwnCellOffsetX", -1);
  m_own_cell_offsets[MD_DirY] = p->getInt64WithDefault("OwnCellOffsetY", -1);
  m_own_cell_offsets[MD_DirZ] = p->getInt64WithDefault("OwnCellOffsetZ", -1);

  m_nb_sub_domains[MD_DirX] = p->getInt32WithDefault("NbSubDomainX", -1);
  m_nb_sub_domains[MD_DirY] = p->getInt32WithDefault("NbSubDomainY", -1);
  m_nb_sub_domains[MD_DirZ] = p->getInt32WithDefault("NbSubDomainZ", -1);

  m_first_own_cell_unique_id = p->getInt64WithDefault("CartesianFirstOnwCellUniqueId", -1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICartesianMeshGenerationInfo* ICartesianMeshGenerationInfo::
getReference(IMesh* mesh,bool create)
{
  //TODO: faire lock pour multi-thread
  const char* name = "CartesianMeshGenerationInfo";
  IUserDataList* udlist = mesh->userDataList();

  IUserData* ud = udlist->data(name,true);
  if (!ud){
    if (!create)
      return nullptr;
    ICartesianMeshGenerationInfo* cm = new impl::CartesianMeshGenerationInfo(mesh);
    udlist->setData(name,new AutoDestroyUserData<ICartesianMeshGenerationInfo>(cm));
    return cm;
  }
  auto* adud = dynamic_cast<AutoDestroyUserData<ICartesianMeshGenerationInfo>*>(ud);
  if (!adud)
    ARCANE_FATAL("Can not cast to ICartesianMeshGenerationInfo");
  return adud->data();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
