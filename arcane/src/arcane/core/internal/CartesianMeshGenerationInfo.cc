// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
{
  m_global_nb_cell = Int64ArrayView(NB_DIM, m_global_nb_cell_ptr);
  m_sub_domain_offset = Int32ArrayView(NB_DIM, m_sub_domain_offset_ptr);
  m_own_nb_cell = Int32ArrayView(NB_DIM, m_own_nb_cell_ptr);
  m_own_cell_offset = Int64ArrayView(NB_DIM, m_own_cell_offset_ptr);

  m_global_nb_cell.fill(-1);
  m_sub_domain_offset.fill(-1);
  m_own_nb_cell.fill(-1);
  m_own_cell_offset.fill(-1);

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setOwnCellOffset(Int64 x,Int64 y,Int64 z)
{
  m_own_cell_offset[0] = x;
  m_own_cell_offset[1] = y;
  m_own_cell_offset[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt64("OwnCellOffsetX", x);
  p->setInt64("OwnCellOffsetY", y);
  p->setInt64("OwnCellOffsetZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setGlobalNbCell(Int64 x,Int64 y,Int64 z)
{
  m_global_nb_cell[0] = x;
  m_global_nb_cell[1] = y;
  m_global_nb_cell[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt64("GlobalNbCellX", x);
  p->setInt64("GlobalNbCellY", y);
  p->setInt64("GlobalNbCellZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setSubDomainOffset(Int32 x,Int32 y,Int32 z)
{
  m_sub_domain_offset[0] = x;
  m_sub_domain_offset[1] = y;
  m_sub_domain_offset[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt32("SubDomainOffsetX", x);
  p->setInt32("SubDomainOffsetY", y);
  p->setInt32("SubDomainOffsetZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
setOwnNbCell(Int32 x,Int32 y,Int32 z)
{
  m_own_nb_cell[0] = x;
  m_own_nb_cell[1] = y;
  m_own_nb_cell[2] = z;

  Properties* p = m_mesh->properties();
  p->setInt32("OwnNbCellX", x);
  p->setInt32("OwnNbCellY", y);
  p->setInt32("OwnNbCellZ", z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshGenerationInfo::
_init()
{
  Properties* p = m_mesh->properties();
  m_global_nb_cell[MD_DirX] = p->getInt64WithDefault("GlobalNbCellX",-1);
  m_global_nb_cell[MD_DirY] = p->getInt64WithDefault("GlobalNbCellY",-1);
  m_global_nb_cell[MD_DirZ] = p->getInt64WithDefault("GlobalNbCellZ",-1);

  m_own_nb_cell[MD_DirX] = p->getInt32WithDefault("OwnNbCellX",-1);
  m_own_nb_cell[MD_DirY] = p->getInt32WithDefault("OwnNbCellY",-1);
  m_own_nb_cell[MD_DirZ] = p->getInt32WithDefault("OwnNbCellZ",-1);

  m_sub_domain_offset[MD_DirX] = p->getInt32WithDefault("SubDomainOffsetX",-1);
  m_sub_domain_offset[MD_DirY] = p->getInt32WithDefault("SubDomainOffsetY",-1);
  m_sub_domain_offset[MD_DirZ] = p->getInt32WithDefault("SubDomainOffsetZ",-1);

  m_own_cell_offset[MD_DirX] = p->getInt64WithDefault("OwnCellOffsetX",-1);
  m_own_cell_offset[MD_DirY] = p->getInt64WithDefault("OwnCellOffsetY",-1);
  m_own_cell_offset[MD_DirZ] = p->getInt64WithDefault("OwnCellOffsetZ",-1);
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
