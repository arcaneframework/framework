// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMng.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Interface du gestionnaire des matériaux et milieux d'un maillage.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshMaterialMng.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IMesh.h"
#include "arcane/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace {
IMeshMaterialMng::IFactory* global_mesh_material_mng_factory = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IMeshMaterialMng::
_internalSetFactory(IFactory* f)
{
  global_mesh_material_mng_factory = f;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMeshMaterialMng> IMeshMaterialMng::
getTrueReference(const MeshHandle& mesh_handle,bool is_create)
{
  auto* f = global_mesh_material_mng_factory;
  if (!f){
    if (is_create)
      ARCANE_FATAL("No factory for 'IMeshMaterialMng': You need to link with 'arcane_materials' library");
    return {};
  }
  return f->getTrueReference(mesh_handle,is_create);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialMng* IMeshMaterialMng::
getReference(const MeshHandle& mesh_handle,bool create)
{
  return getTrueReference(mesh_handle,create).get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialMng* IMeshMaterialMng::
getReference(IMesh* mesh,bool is_create)
{
  ARCANE_CHECK_POINTER(mesh);
  return getReference(mesh->handle(),is_create);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
