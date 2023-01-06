// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshIntersectionUnitTest.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Service de test des variables.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParticleFamily.h"
#include "arcane/IRayMeshIntersection.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemVector.h"
#include "arcane/ServiceBuilder.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/RayMeshIntersectionUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des variables
 */
class RayMeshIntersectionUnitTest
: public ArcaneRayMeshIntersectionUnitTestObject
{
 public:

  explicit RayMeshIntersectionUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  void _testReferences(Integer nb_ref);
  void _testUsed();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_RAYMESHINTERSECTIONUNITTEST(RayMeshIntersectionUnitTest,
                                                    RayMeshIntersectionUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RayMeshIntersectionUnitTest::
RayMeshIntersectionUnitTest(const ServiceBuildInfo& mb)
: ArcaneRayMeshIntersectionUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RayMeshIntersectionUnitTest::
executeTest()
{
  IParallelMng* pm = mesh()->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  info() << "EXEC TEST rank=" << my_rank << " nb_rank=" << nb_rank;
  // Creation de n rayons
  Integer n0 = 15; // 40 pour un plus gros test.
  Integer nb_segment = (n0*n0) / nb_rank;
  Real3UniqueArray segments_position(nb_segment);
  Real3UniqueArray segments_direction(nb_segment);
  Real3UniqueArray segments_intersection(nb_segment);
  RealUniqueArray segments_distance(nb_segment);
  Int32UniqueArray segments_user_value(nb_segment);
  Int32UniqueArray segments_orig_face(nb_segment,NULL_ITEM_LOCAL_ID);
  for( Integer i=0; i<nb_segment; ++i ){
    Integer index = i +  nb_segment*my_rank;
    const Int32 ipx = index / n0;
    const Int32 ipy = index % n0;
    Real px = static_cast<Real>(ipx);
    Real py = static_cast<Real>(ipy);
    segments_position[i] = Real3(7.0,py/10.0-py/5.0,px/10.0-px/5.0);
    segments_direction[i] = Real3(-1.0-(px/5.0),(py/20.0)-(py/10.0),0.5-(px/10.0));
    info() << " SEGMENT: pos=" << segments_position[i] << " dir=" << segments_direction[i];

  }
  Int32UniqueArray faces_lid(nb_segment);
  //IRayMeshIntersection* mi = createBasicRayMeshIntersection(mesh());
  
  //IServiceMng* sm = subDomain()->serviceMng();
  //FactoryT<IRayMeshIntersection> factory(sm);
  ServiceBuilder<IRayMeshIntersection> sb(subDomain());
  auto mi(sb.createReference("BasicRayMeshIntersection"));

  // Calcul avec les segments
  mi->compute(segments_position,segments_direction,segments_orig_face,segments_user_value,
              segments_intersection,segments_distance,faces_lid);

  // Calcul avec les rayons sous forme de particule
  IItemFamily* ray_family = mesh()->findItemFamily(IK_Particle,"Rays",true);
  Int64UniqueArray uids(nb_segment);
  Int32UniqueArray local_ids(nb_segment);
  for( Integer i=0; i<nb_segment; ++i )
    uids[i] = (i+(nb_segment*my_rank)+1);
  IParticleFamily* pf = ray_family->toParticleFamily();
  pf->addParticles(uids,local_ids);
  ray_family->endUpdate();
  VariableParticleReal3 rays_position(VariableBuildInfo(ray_family,"Position"));
  VariableParticleReal3 rays_direction(VariableBuildInfo(ray_family,"Direction"));
  VariableParticleReal3 rays_intersection(VariableBuildInfo(ray_family,"Intersection"));
  VariableParticleReal rays_distance(VariableBuildInfo(ray_family,"Distance"));
  VariableParticleInt32 user_values(VariableBuildInfo(ray_family,"UserValues"));
  VariableParticleInt32 rays_orig_face(VariableBuildInfo(ray_family,"OriginalFace"));
  VariableParticleInt32 rays_face(VariableBuildInfo(ray_family,"Face"));
  // Pour test, supprime quelques rayons
  {
    Int32UniqueArray to_remove_rays;
    for( Integer i=0; i<(nb_segment/10); ++i ){
      to_remove_rays.add(i*5);
    }
    pf->removeParticles(to_remove_rays);
    pf->endUpdate();
  }
  
  ENUMERATE_PARTICLE(iitem,ray_family->allItems()){
    Particle p = *iitem;
    pf->setParticleCell(p,Cell());
    rays_position[p] = segments_position[p.localId()];
    rays_direction[p] = segments_direction[p.localId()];
    rays_orig_face[p] = NULL_ITEM_LOCAL_ID;
  }
  mi->compute(ray_family,rays_position,rays_direction,rays_orig_face,user_values,
              rays_intersection,rays_distance,rays_face);
  info() << "Print rays infos";
  FaceInfoListView faces_internal(mesh()->faceFamily());
  ENUMERATE_PARTICLE(ipart,ray_family->allItems()){
    Particle ray = *ipart;
    Int32 face_lid = rays_face[ipart];
    if (face_lid!=NULL_ITEM_ID)
      info() << "Ray uid=" << ray.uniqueId()
             << " pos=" << rays_position[ipart]
             << " dir=" << rays_direction[ipart]
             << " intersect_face_lid=" << face_lid
             << " intersect_face_uid=" << faces_internal[face_lid]->uniqueId()
             << " d=" << rays_distance[ipart]
             << " p=" << rays_intersection[ipart];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RayMeshIntersectionUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
