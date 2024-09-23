// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCaseMeshMasterService.h                               (C) 2000-2024 */
/*                                                                           */
/* Service Arcane gérant les maillages du jeu de données.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/ICaseMeshMasterService.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/ICaseMeshService.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/impl/ArcaneCaseMeshMasterService_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneCaseMeshMasterService
: public ArcaneArcaneCaseMeshMasterServiceObject
{
 public:

  explicit ArcaneCaseMeshMasterService(const ServiceBuildInfo& sbi)
  : ArcaneArcaneCaseMeshMasterServiceObject(sbi)
  , m_sub_domain(sbi.subDomain())
  {}

 private:

  void createMeshes() override
  {
    if (m_is_created)
      ARCANE_FATAL("Meshes have already been created");

    Integer nb_mesh = options()->mesh.size();
    info() << "Creating meshes from 'ArcaneCaseMeshMasterService' nb_mesh=" << nb_mesh;
    Integer index = 0;
    for( ICaseMeshService* s : options()->mesh ){
      String name = String("Mesh")+String::fromNumber(index);
      s->createMesh(name);
      ++index;
    }
    m_is_created = true;
  }

  void allocateMeshes() override
  {
    if (m_is_allocated)
      ARCANE_FATAL("Meshes have already been allocated");
    if (!m_is_created)
      ARCANE_FATAL("You need to call createMeshes() before allocateMeshes()");

    for( ICaseMeshService* s : options()->mesh )
      s->allocateMeshItems();
    m_is_allocated = true;
  }

  void partitionMeshes() override
  {
    if (!m_is_allocated)
      ARCANE_FATAL("Meshes have do be allocated before partitioning. call allocateMeshes() before");
    for( ICaseMeshService* s : options()->mesh )
      s->partitionMesh();
  }

  void applyAdditionalOperationsOnMeshes() override
  {
    if (!m_is_allocated)
      ARCANE_FATAL("Meshes are not allocated. call allocateMeshes() before");
    for (ICaseMeshService* s : options()->mesh)
      s->applyAdditionalOperations();
  }

  ICaseOptions* _options() override
  {
    return options()->caseOptions();
  }
 private:
  ISubDomain* m_sub_domain;
  bool m_is_created = false;
  bool m_is_allocated = false;
  UniqueArray<IPrimaryMesh*> m_meshes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANECASEMESHMASTERSERVICE(ArcaneCaseMeshMasterService,
                                                    ArcaneCaseMeshMasterService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
