// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshMeshGenerationInfo.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Informations d'un maillage issu du format 'msh'.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/MshMeshGenerationInfo.h"

#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/AutoDestroyUserData.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshGenerationInfo::
MshMeshGenerationInfo(IMesh* mesh)
: m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MshMeshGenerationInfo* MshMeshGenerationInfo::
getReference(IMesh* mesh, bool create)
{
  const char* name = "MshMeshGenerationInfo";
  IUserDataList* udlist = mesh->userDataList();

  IUserData* ud = udlist->data(name, true);
  if (!ud) {
    if (!create)
      return nullptr;
    auto* cm = new MshMeshGenerationInfo(mesh);
    udlist->setData(name, new AutoDestroyUserData<MshMeshGenerationInfo>(cm));
    return cm;
  }
  auto* adud = dynamic_cast<AutoDestroyUserData<MshMeshGenerationInfo>*>(ud);
  if (!adud)
    ARCANE_FATAL("Can not cast to MshMeshGenerationInfo");
  return adud->data();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
