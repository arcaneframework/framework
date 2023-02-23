// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMeshService.cc                        (C) 2000-2021             */
/*                                                                           */
/* WIP : first POC for general mesh. Not yet plugged                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshBuildInfo.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/ICaseMeshService.h"
#include "arcane/mesh/PolyhedralMesh.h"

#include "PolyhedralMesh_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PolyhedralMeshService : public ArcanePolyhedralMeshObject
{
 public:

  explicit PolyhedralMeshService(const ServiceBuildInfo& sbi)
  : ArcanePolyhedralMeshObject(sbi)
  {}

 public:

  void createMesh(const String& name) override
  {
    info() << "---CREATE MESH---- " << name;
    info() << "--Read mesh file " << options()->file();
    mesh.read(options()->file);
  }

  void allocateMeshItems() override
  {
    info() << "---ALLOCATE MESH ITEMS---- ";
  }

  void partitionMesh() override {}

 private:

  mesh::PolyhedralMesh mesh{ subDomain(), MeshBuildInfo{ "DefaultMesh" } };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_POLYHEDRALMESH(PolyhedralMesh, PolyhedralMeshService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/