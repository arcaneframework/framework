// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicRayMeshIntersection.cc                                 (C) 2000-2015 */
/*                                                                           */
/* Service basique de calcul d'intersection entre segments et maillage.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/BasicService.h"
#include "arcane/FactoryService.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ItemGroup.h"
#include "arcane/VariableTypes.h"
#include "arcane/IRayMeshIntersection.h"
#include "arcane/IParallelMng.h"

#include <Lima/lima++.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul l'intersection d'un rayon avec un ensemble de triangles en 3D.
 *
 * Un rayon est une demi-droite et est défini par son origine et sa direction.
 * Il faut positionner les rayons (via setRays()) et la liste des triangles
 * (via setTriangles()) puis appeler la méthode compute(). En retour, on
 * peut récupérer pour chaque rayon la distance (distances()) et le triangle intersecté
 * (intersectedTriangleIndexes()).
 *
 * Les vues passées en argument (setRays() et setTriangles()) ne doivent pas être
 * modifiées tant que l'instance existe.
 */
class RayTriangle3DIntersection
: public TraceAccessor
{
 public:
  RayTriangle3DIntersection(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  /*!
   * \brief Position la liste des rayons dont on souhaite calculer l'intersection.
   */
  void setRays(Real3ConstArrayView origins,Real3ConstArrayView directions)
  {
    m_rays_origin = origins;
    m_rays_direction = directions;
  }
  /*!
   * \brief Positionne la liste des triangles dont on souhaite calculer l'intersection
   * avec les rayons. Le tableau \a indexes contient pour chaque triangle les indices
   * dans le tableau \a coordinates de chaque sommet. Par exemple,
   * indexes[0..2] contient les indices des sommets du 1er triangle, indexes[3..5]
   * ceux du second.
   */
  void setTriangles(Real3ConstArrayView coordinates,Int32ConstArrayView indexes)
  {
    m_triangles_coordinates = coordinates;
    m_triangles_indexes = indexes;
  }
  /*!
   * \brief Calcul l'intersection de chaque rayon avec la liste des triangles.
   * Si un rayon intersecte plusieurs triangles, on concerve celui dont
   * la distance par rapport à l'origine du rayon est la plus petite.
   */
  void compute();
  /*!
   * \brief Distance de l'origine d'un rayon à son point d'intersection.
   * Distance (exprimée relativivement à la norme de \a directions)
   * du point d'intersection d'un rayon par rapport à son origine.
   * Pour le rayon \a i, son point d'intersection est donc donnée
   * par la formule (origins[i] + distances[i]*directions[i]).
   * La distance est négative si le rayon n'intersecte aucun triangle.
   * Ce tableau est remplit lors de l'appel à compute().
   */
  RealConstArrayView distances()
  {
    return m_distances;
  }
  
  /*!
   * \brief Indices des triangles intersectés.
   * Indice dans le tableau donnée par \a setTriangles() du triangle
   * intersecté par chaque rayon. Cet indice vaut (-1) si un rayon
   * n'intersecte pas un triangle. Ce tableau est remplit lors
   * de l'appel à compute().
   */
  Int32ConstArrayView intersectedTriangleIndexes()
  {
    return m_intersected_triangle_indexes;
  }

  Real checkIntersection(Real3 origin,Real3 direction,Real3 p0,Real3 p1,Real3 p2);

  static bool checkBoundingBox(Real3 origin,Real3 direction,Real3 box_min,Real3 box_max);

 private:

  void _compute2(Int32 triangle_id,Real3 p0,Real3 p1,Real3 p2);

 private:
  
  Real3ConstArrayView m_rays_origin;
  Real3ConstArrayView m_rays_direction;
  Real3ConstArrayView m_triangles_coordinates;
  Int32ConstArrayView m_triangles_indexes;

  RealUniqueArray m_distances;
  Int32UniqueArray m_intersected_triangle_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RayTriangle3DIntersection::
compute()
{
  Integer nb_ray = m_rays_origin.size();
  m_distances.resize(nb_ray);
  m_distances.fill(1.0e100);
  m_intersected_triangle_indexes.resize(nb_ray);
  m_intersected_triangle_indexes.fill(-1);
  Integer nb_index = m_triangles_indexes.size();
  if ((nb_index%3)!=0)
    throw FatalErrorException(A_FUNCINFO,"bad triangle_index count (({0}) % 3)!=0");
  Integer nb_triangle = nb_index / 3;
  info() << "COMPUTE RAY INTERSECTION nb_ray=" << nb_ray
         << " nb_triangle=" << nb_triangle;

  for( Integer i=0; i<nb_triangle; ++i ){
    Real3 p0 = m_triangles_coordinates[m_triangles_indexes[(i*3)]];
    Real3 p1 = m_triangles_coordinates[m_triangles_indexes[(i*3)+1]];
    Real3 p2 = m_triangles_coordinates[m_triangles_indexes[(i*3)+2]];
    _compute2(i,p0,p1,p2);
  }

  for( Integer i=0; i<nb_ray; ++i ){
    Int32 tid = m_intersected_triangle_indexes[i];
    if (tid==(-1))
      m_distances[i] = -1.0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RayTriangle3DIntersection::
_compute2(Int32 triangle_id,Real3 p0,Real3 p1,Real3 p2)
{
  Integer nb_ray = m_rays_origin.size();
  for( Integer i=0; i<nb_ray; ++i ){
    Real t = checkIntersection(m_rays_origin[i],m_rays_direction[i],p0,p1,p2);
    if (t>=0.0){
      info() << "Segment " << i << " intersect triangle " << triangle_id << " T=" << t;
      if (t<m_distances[i]){
        m_distances[i] = t;
        m_intersected_triangle_indexes[i] = triangle_id;
        info() << "Get this triangle";
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul l'intersection de la demi-droite [origin,direction)
 * avec le triangle (p0,p1,p2).
 *
 * La direction n'a pas besoin d'être normalisée.
 *
 * La position du point d'intersection est P = origin + t * direction
 * où \a t est la valeur retournée par cette fonction. Cette valeur
 * est négative si si aucun point d'intersection n'est trouvé.
 */
Real RayTriangle3DIntersection::
checkIntersection(Real3 origin,
                  Real3 direction,
                  Real3 p0,Real3 p1,Real3 p2)
{
  // Cette routine s'inspire du code de la bibliothèque WildMagic
  // dont la licence est ci-dessous.
  
  // Wild Magic Source Code
  // David Eberly
  // http://www.geometrictools.com
  // Copyright (c) 1998-2009
  //
  // This library is free software; you can redistribute it and/or modify it
  // under the terms of the GNU Lesser General Public License as published by
  // the Free Software Foundation; either version 2.1 of the License, or (at
  // your option) any later version.  The license is available for reading at
  // either of the locations:
  //     http://www.gnu.org/copyleft/lgpl.html
  //     http://www.geometrictools.com/License/WildMagicLicense.pdf
  //


  Real ZERO_TOLERANCE = 1.0e-14;
  // compute the offset origin, edges, and normal
  Real3 kDiff = origin - p0;
  Real3 kEdge1 = p1 - p0;
  Real3 kEdge2 = p2 - p0;
  Real3 kNormal = math::vecMul(kEdge1,kEdge2);
    
  Real fDdN = math::dot(direction,kNormal);
  Real fSign;
  if (fDdN > ZERO_TOLERANCE)
  {
    fSign = (Real)1.0;
  }
  else if (fDdN < -ZERO_TOLERANCE)
  {
    fSign = (Real)-1.0;
    fDdN = -fDdN;
  }
  else
  {
    // Segment and triangle are parallel, call it a "no intersection"
    // even if the segment does intersect.
    return (-1.0);
  }

  Real fDdQxE2 = fSign*math::dot(direction,math::vecMul(kDiff,kEdge2));
  if (fDdQxE2 >= (Real)0.0)
  {
    Real fDdE1xQ = fSign*math::dot(direction,math::vecMul(kEdge1,kDiff));
    if (fDdE1xQ >= (Real)0.0)
    {
      Real diff = fDdN-(fDdQxE2 + fDdE1xQ);
      // L'epsilon sert si le segment traverse par la face à proximité
      // d'une des arêtes de la face. Dans ce cas, on considère qu'il
      // y a intersection. Sans cela, à cause des arrondis numériques,
      // un segment pourrait traverser le maillage en passant par une
      // arête entre deux faces sans intersecter ces dernières.
      if (diff>=0.0 || diff>-ZERO_TOLERANCE)
      {
        // line intersects triangle, check if segment does
        Real fQdN = -fSign*math::dot(kDiff,kNormal);
        
        // Comme il s'agit d'une demi-droite et pas d'un segment,
        // il y a toujours intersection si \a t est positif.
#if 0
        Real fExtDdN = m_pkSegment->Extent*fDdN;
        if (-fExtDdN <= fQdN && fQdN <= fExtDdN)
        {
          // segment intersects triangle
          return true;
        }
#endif
        Real fInv = ((Real)1.0)/fDdN;
        Real t = fQdN*fInv;
        //Real b1 = fDdQxE2*fInv;
        //Real b2 = fDdE1xQ*fInv;
        //Real b0 = (Real)1.0 - b1 - b2;
        //   P = origin + t*direction = b0*V0 + b1*V1 + b2*V2
        //info() << "** INTERSECT POS=" << b0 << " y=" << b1 << " z=" << b2 << " T=" << t;
        //Real intersect_p = origin + t*direction;
        return t;

        // else: |t| > extent, no intersection
      }
      // else: b1+b2 > 1, no intersection
    }
    // else: b2 < 0, no intersection
  }
  // else: b1 < 0, no intersection

  return (-1.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RayTriangle3DIntersection::
checkBoundingBox(Real3 origin,Real3 direction,Real3 box_min,Real3 box_max)
{
  Real3 box_center = (box_max+box_min) * 0.5;
  Real afWdU[3], afAWdU[3], afDdU[3], afADdU[3], afAWxDdU[3], fRhs;

  Real3 kDiff = origin - box_center;
  Real3 Axis[3];
  //Axis[0].x = box_max.x - box_min.x;
  //Axis[1].y = box_max.y - box_min.y;
  //Axis[2].z = box_max.z - box_min.z;
  Axis[0] = Real3(1.0,0.0,0.0);
  Axis[1] = Real3(0.0,1.0,0.0);
  Axis[2] = Real3(0.0,0.0,1.0);
  Real Extent[3];
  Extent[0] = box_max.x - box_min.x;
  Extent[1] = box_max.y - box_min.y;
  Extent[2] = box_max.z - box_min.z;

  afWdU[0] = math::dot(direction,Axis[0]);
  afAWdU[0] = math::abs(afWdU[0]);
  afDdU[0] = math::dot(kDiff,Axis[0]);
  afADdU[0] = math::abs(afDdU[0]);
  if (afADdU[0] > Extent[0] && afDdU[0]*afWdU[0] >= (Real)0.0)
  {
    return false;
  }

  afWdU[1] = math::dot(direction,Axis[1]);
  afAWdU[1] = math::abs(afWdU[1]);
  afDdU[1] = math::dot(kDiff,Axis[1]);
  afADdU[1] = math::abs(afDdU[1]);
  if (afADdU[1] > Extent[1] && afDdU[1]*afWdU[1] >= (Real)0.0)
  {
    return false;
  }

  afWdU[2] = math::dot(direction,Axis[2]);
  afAWdU[2] = math::abs(afWdU[2]);
  afDdU[2] = math::dot(kDiff,Axis[2]);
  afADdU[2] = math::abs(afDdU[2]);
  if (afADdU[2] > Extent[2] && afDdU[2]*afWdU[2] >= (Real)0.0)
  {
    return false;
  }

  Real3 kWxD = math::vecMul(direction,kDiff);

  afAWxDdU[0] = math::abs(math::dot(kWxD,Axis[0]));
  fRhs = Extent[1]*afAWdU[2] + Extent[2]*afAWdU[1];
  if (afAWxDdU[0] > fRhs)
  {
    return false;
  }

  afAWxDdU[1] = math::abs(math::dot(kWxD,Axis[1]));
  fRhs = Extent[0]*afAWdU[2] + Extent[2]*afAWdU[0];
  if (afAWxDdU[1] > fRhs)
  {
    return false;
  }

  afAWxDdU[2] = math::abs(math::dot(kWxD,Axis[2]));
  fRhs = Extent[0]*afAWdU[1] + Extent[1]*afAWdU[0];
  if (afAWxDdU[2] > fRhs)
  {
    return false;
  }

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicRayFaceIntersector
: public IRayFaceIntersector
{
 public:
  BasicRayFaceIntersector(ITraceMng* tm)
  : m_triangle_intersector(tm)
  {
  }
 public:
  bool computeIntersection(Real3 origin,Real3 direction,
                           Int32 orig_face_local_id,
                           Int32 face_local_id,
                           Real3ConstArrayView face_nodes,
                           Int32* user_value,
                           Real* distance,Real3* position)
  {
    ARCANE_UNUSED(orig_face_local_id);
    ARCANE_UNUSED(face_local_id);

    Integer nb_node = face_nodes.size();
    switch(nb_node){
    case 4:
      {
        Real3 center = (face_nodes[0] + face_nodes[1] + face_nodes[2] + face_nodes[3]) / 4.0;
        Real d0 = m_triangle_intersector.checkIntersection(origin,direction,center,face_nodes[0],face_nodes[1]);
        Real d1 = m_triangle_intersector.checkIntersection(origin,direction,center,face_nodes[1],face_nodes[2]);
        Real d2 = m_triangle_intersector.checkIntersection(origin,direction,center,face_nodes[2],face_nodes[3]);
        Real d3 = m_triangle_intersector.checkIntersection(origin,direction,center,face_nodes[3],face_nodes[0]);
        Real d = 1.0e100;
        bool found = false;
        if (d0>=0.0 && d0<d){
          found = true;
          d = d0;
        }
        if (d1>=0.0 && d1<d){
          found = true;
          d = d1;
        }
        if (d2>=0.0 && d2<d){
          found = true;
          d = d2;
        }
        if (d3>=0.0 && d3<d){
          found = true;
          d = d3;
        }
        if (!found)
          d = -1.0;
        *distance = d;
        *position = origin + d * direction;
        *user_value = 1;
        return found;
      }
      break;
    default:
      throw NotImplementedException(A_FUNCINFO,"only quad face is implemented");
    }
  }
 public:
  RayTriangle3DIntersection m_triangle_intersector;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service basique de calcul d'intersection entre segments et maillage.
 */
class BasicRayMeshIntersection
: public BasicService
, public IRayMeshIntersection
{
 public:

  BasicRayMeshIntersection(const ServiceBuildInfo& sbi);
  virtual ~BasicRayMeshIntersection() {}

 public:

  virtual void build(){}
  virtual void compute(Real3ConstArrayView segments_position,
                       Real3ConstArrayView segments_direction,
                       Int32ConstArrayView segments_orig_face,
                       Int32ArrayView user_values,
                       Real3ArrayView intersections,
                       RealArrayView distances,
                       Int32ArrayView faces_local_id);

  virtual void compute(IItemFamily* ray_family,
                       VariableParticleReal3& rays_position,
                       VariableParticleReal3& rays_direction,
                       VariableParticleInt32& rays_orig_face,
                       VariableParticleInt32& user_values,
                       VariableParticleReal3& intersections,
                       VariableParticleReal& distances,
                       VariableParticleInt32& rays_face);

  virtual void setFaceIntersector(IRayFaceIntersector* intersector)
  {
    m_face_intersector = intersector;
  }
  virtual IRayFaceIntersector* faceIntersector()
  {
    return m_face_intersector;
  }
	
 public:

  inline void _checkBoundingBox(Real3 p,Real3* ARCANE_RESTRICT min_bounding_box,
                         Real3* ARCANE_RESTRICT max_bounding_box)
  {
    if (p.x<min_bounding_box->x)
      min_bounding_box->x = p.x;
    if (p.y<min_bounding_box->y)
      min_bounding_box->y = p.y;
    if (p.z<min_bounding_box->z)
      min_bounding_box->z = p.z;

    if (p.x>max_bounding_box->x)
      max_bounding_box->x = p.x;
    if (p.y>max_bounding_box->y)
      max_bounding_box->y = p.y;
    if (p.z>max_bounding_box->z)
      max_bounding_box->z = p.z;
  }

  void _writeSegments(Int32 rank,
                      Real3ConstArrayView positions,
                      Real3ConstArrayView directions,
                      RealConstArrayView distances);
 private:
  IRayFaceIntersector* m_face_intersector;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicRayMeshIntersection::
BasicRayMeshIntersection(const ServiceBuildInfo& sbi)
: BasicService(sbi)
, m_face_intersector(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicRayMeshIntersection::
compute(Real3ConstArrayView segments_position,
        Real3ConstArrayView segments_direction,
        Int32ConstArrayView segments_orig_face,
        Int32ArrayView user_values,
        Real3ArrayView segments_intersection,
        RealArrayView segments_distance,
        Int32ArrayView faces_local_id)
{
  IMesh* mesh = this->mesh();
  
  bool is_3d = mesh->dimension()==3;
  info() << "COMPUTE INTERSECTION!!";
  FaceGroup outer_faces = mesh->outerFaces();
  Integer nb_face = outer_faces.size();
  Integer nb_segment = segments_position.size();
  info() << "NB OUTER FACE=" << nb_face << " NB_SEGMENT=" << nb_segment;
  VariableNodeReal3 nodes_coordinates(mesh->nodesCoordinates());
  const Real max_value = 1.0e100;
  const Real3 max_bb(max_value,max_value,max_value);
  const Real3 min_bb(-max_value,-max_value,-max_value);

  Real3 mesh_min_bounding_box(max_bb);
  Real3 mesh_max_bounding_box(min_bb);
  Integer max_face_local_id = mesh->faceFamily()->maxLocalId();
  Real3UniqueArray faces_min_bb(max_face_local_id);
  Real3UniqueArray faces_max_bb(max_face_local_id);

  // Calcule les bounding box
  {
    ENUMERATE_FACE(iface,outer_faces){
      Int32 lid = iface.localId();
      Real3 face_max_bb(max_bb);
      Real3 face_min_bb(min_bb);
      for( NodeEnumerator inode((*iface).nodes()); inode.hasNext(); ++inode )
        _checkBoundingBox(nodes_coordinates[inode],&face_min_bb,&face_max_bb);
      _checkBoundingBox(face_min_bb,&mesh_min_bounding_box,&mesh_max_bounding_box);
      _checkBoundingBox(face_max_bb,&mesh_min_bounding_box,&mesh_max_bounding_box);
      //TODO: peut-etre ajouter un epsilon autour de le BB pour eviter
      // les erreurs d'arrondi dans la determination de l'intersection
      faces_min_bb[lid] = face_min_bb;
      faces_max_bb[lid] = face_max_bb;
    }
  }

  //RealArray segments_distance;
  segments_distance.fill(max_value);
  faces_local_id.fill(NULL_ITEM_LOCAL_ID);

  if (!m_face_intersector)
    m_face_intersector = new BasicRayFaceIntersector(traceMng());

  Real3UniqueArray face_nodes;
  for( Integer i=0; i<nb_segment; ++i ){
    Real3 position = segments_position[i];
    Real3 direction = segments_direction[i];
    Int32 orig_face_local_id = segments_orig_face[i];
    Real3 intersection;
    Real distance = 1.0e100;
    Int32 min_face_local_id = NULL_ITEM_LOCAL_ID;
    Int32 user_value = 0;
    ENUMERATE_FACE(iface,outer_faces){
      const Face& face = *iface;
      // On ne traite que ses propres faces
      if (!face.isOwn())
        continue;
      Int32 lid = face.localId();
      // En 3D, cherche si intersection avec la bounding box.
      // s'il n'y en a pas, inutile d'aller plus loin.
      bool is_bb = true;
      if (is_3d)
        is_bb = RayTriangle3DIntersection::checkBoundingBox(position,direction,faces_min_bb[lid],faces_max_bb[lid]);
      if (!is_bb)
        continue;
      Integer nb_node = face.nbNode();
      face_nodes.resize(nb_node);
      for( NodeEnumerator inode(face.nodes()); inode.hasNext(); ++inode )
        face_nodes[inode.index()] = nodes_coordinates[inode];
      Real d = 0.0;
      Real3 local_intersection;
      Int32 uv = 0;
      bool is_found = m_face_intersector->computeIntersection(position,direction,orig_face_local_id,lid,
                                                              face_nodes,&uv,&d,&local_intersection);
      //if (i==15){
      //  for( Integer z=0; z<nb_node; ++z )
      //    info() << "FACE NODE lid=" << lid << " I=" << i << " v=" << face_nodes[z] << " d=" << d;
      //}
      //if (!is_bb)
        //info() << "CHECK INTERSECTION: is_bb=" << is_bb << " " << is_found << '\n';
      if (is_found && !is_bb)
        fatal() << "Intersection found but no bounding box intersection";
      if (is_found && d<distance){
        distance = d;
        min_face_local_id = lid;
        intersection = local_intersection;
        user_value = uv;
      }
    }
    if (min_face_local_id==NULL_ITEM_LOCAL_ID)
      distance = -1.0;
    segments_distance[i] = distance;
    segments_intersection[i] = intersection;
    faces_local_id[i] = min_face_local_id;
    user_values[i] = user_value;
  }
}

struct FoundInfo
{
  Int32 contrib_owner;
  Int32 owner;
  Int32 local_id;
  Int32 face_local_id;
  Int32 orig_face_local_id;
  Int32 user_value;
  Real distance;
  Real3POD intersection;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicRayMeshIntersection::
compute(IItemFamily* ray_family,
        VariableParticleReal3& rays_position,
        VariableParticleReal3& rays_direction,
        VariableParticleInt32& rays_orig_face,
        VariableParticleInt32& rays_user_value,
        VariableParticleReal3& intersections,
        VariableParticleReal& distances,
        VariableParticleInt32& rays_face)
{
  // NOTE: rays_orig_face n'est pas utilisé.

  IMesh* mesh = ray_family->mesh();
  IParallelMng* pm = mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  // Suppose que les rayons sont compactés
  Integer nb_local_ray = ray_family->allItems().size();
  //if (nb_local_ray!=ray_family->maxLocalId()){
  //fatal() << "La famille de rayons doit être compactée nb=" << nb_local_ray
  //          << " max_id=" << ray_family->maxLocalId();
  //}
  Integer global_nb_ray = pm->reduce(Parallel::ReduceSum,nb_local_ray);
  if (global_nb_ray==0)
    return;
  info() << "LOCAL_NB_RAY=" << nb_local_ray << " GLOBAL=" << global_nb_ray;

  // Recopie dans un tableau local les informations d'entrée.
  // Cela est toujours nécessaire car les particules ne sont pas forcément
  // compactées et il peut y avoir des trous dans la numérotation.
  Real3UniqueArray local_positions(nb_local_ray);
  Real3UniqueArray local_directions(nb_local_ray);

  ENUMERATE_ITEM(iitem,ray_family->allItems()){
    Integer index = iitem.index();
    //local_ids[index] = iitem.localId();
    //unique_ids[index] = (*iitem).uniqueId();
    local_positions[index] = rays_position[iitem];
    local_directions[index] = rays_direction[iitem];
  }

  if (pm->isParallel()){

    Int32UniqueArray local_ids(nb_local_ray);
    Int64UniqueArray unique_ids(nb_local_ray);
    ENUMERATE_ITEM(iitem,ray_family->allItems()){
      Integer index = iitem.index();
      local_ids[index] = iitem.localId();
      unique_ids[index] = (*iitem).uniqueId();
    }

    // En parallèle, pour l'instant récupère les rayons des autres processeurs

    Real3UniqueArray all_positions;
    pm->allGatherVariable(local_positions,all_positions);
    Real3UniqueArray all_directions; 
    pm->allGatherVariable(local_directions,all_directions);

    Int32UniqueArray local_owners(nb_local_ray);
    local_owners.fill(my_rank);

    Int32UniqueArray all_owners;
    pm->allGatherVariable(local_owners,all_owners);
    Int32UniqueArray all_local_ids;
    pm->allGatherVariable(local_ids,all_local_ids);
    Int64UniqueArray all_unique_ids;
    pm->allGatherVariable(unique_ids,all_unique_ids);

    Int32UniqueArray all_user_values(global_nb_ray);
    Int32UniqueArray all_orig_faces(global_nb_ray,NULL_ITEM_LOCAL_ID);
    RealUniqueArray all_distances(global_nb_ray);
    Real3UniqueArray all_intersections(global_nb_ray);
    Int32UniqueArray all_faces(global_nb_ray);
    // Mettre à (-1) les faces dont je ne suis pas le propriétaire
    for( Integer z=0, zn=all_orig_faces.size(); z<zn; ++z ){
      if (all_owners[z]!=my_rank)
        all_orig_faces[z] = NULL_ITEM_LOCAL_ID;
    }
    compute(all_positions,all_directions,all_orig_faces,all_user_values,all_intersections,all_distances,all_faces);
    /*for( Integer i=0; i<global_nb_ray; ++i ){
      info() << "RAY I=" << i << " uid=" << all_unique_ids[i] << " lid=" << all_local_ids[i]
             << " owner=" << all_owners[i] << " position=" << all_positions[i] 
             << " direction=" << all_directions[i] << " distance=" << all_distances[i]
             << " face=" << all_faces[i];
             }*/
    {
      UniqueArray<FoundInfo> founds_info;
      
      for( Integer i=0; i<global_nb_ray; ++i ){
        Int32 face_lid = all_faces[i];
        if (face_lid==NULL_ITEM_LOCAL_ID)
          continue;
        FoundInfo fi;
        fi.contrib_owner = my_rank;
        fi.owner = all_owners[i];
        fi.local_id = all_local_ids[i];
        fi.face_local_id = all_faces[i];
        fi.orig_face_local_id = all_orig_faces[i];
        fi.user_value = all_user_values[i];
        fi.intersection = all_intersections[i];
        fi.distance = all_distances[i];
        founds_info.add(fi);
      }
      info() << "NB_FOUND=" << founds_info.size();
      //Array<FoundInfo> global_founds_info;
      ByteUniqueArray global_founds_info_bytes;
      Integer finfo_byte_size = arcaneCheckArraySize(founds_info.size()*sizeof(FoundInfo));
      ByteConstArrayView local_fi(finfo_byte_size,(const Byte*)founds_info.data());
      pm->allGatherVariable(local_fi,global_founds_info_bytes);
      Integer gfi_byte_size = arcaneCheckArraySize(global_founds_info_bytes.size()/sizeof(FoundInfo));
      ConstArrayView<FoundInfo> global_founds_info(gfi_byte_size,(const FoundInfo*)global_founds_info_bytes.data());
      Integer nb_total_found = global_founds_info.size();
      ItemInternalList rays = ray_family->itemsInternal();
      rays_face.fill(NULL_ITEM_LOCAL_ID);
      distances.fill(0.0);
      VariableItemInt32& rays_new_owner = ray_family->itemsNewOwner();
      rays_new_owner.fill(my_rank);
      for( Integer i=0; i<nb_total_found; ++i ){
        const FoundInfo& fi = global_founds_info[i];
        Int32 owner = fi.owner;
        // Il ne faut traiter que ses propres rayons
        if (owner!=my_rank)
          continue;
        Int32 local_id = fi.local_id;
        Particle ray = rays[local_id];
        Real distance = fi.distance;
        if ((rays_face[ray]==NULL_ITEM_LOCAL_ID) || distance<distances[ray]){
          distances[ray] = distance;
          intersections[ray] = Real3(fi.intersection.x,fi.intersection.y,fi.intersection.z);
          rays_face[ray] = fi.face_local_id;
          rays_user_value[ray] = fi.user_value;
          rays_new_owner[ray] = fi.contrib_owner;
          //info() << "SET RAY uid=" << ray.uniqueId() << " distance=" << distance << " new_owner=" << fi.contrib_owner;
        }
        /*info() << "GLOBAL RAY I=" << i << " uid=" << ray.uniqueId() << " lid=" << fi.local_id
               << " owner=" << fi.owner
               << " distance=" << fi.distance
               << " contrib_owner=" << fi.contrib_owner
               << " face=" << fi.face_local_id
               << " new_owner=" << rays_new_owner[ray];*/
      }
      ray_family->exchangeItems();
    }
  }
  else{

    Int32UniqueArray orig_faces(nb_local_ray);
    Int32UniqueArray user_values(nb_local_ray);

    ENUMERATE_ITEM(iitem,ray_family->allItems()){
      Integer index = iitem.index();
      orig_faces[index] = rays_orig_face[iitem];
      user_values[index] = rays_user_value[iitem];
    }

    Real3UniqueArray out_intersections(nb_local_ray);
    RealUniqueArray out_distances(nb_local_ray);
    Int32UniqueArray out_faces(nb_local_ray);

    compute(local_positions,local_directions,orig_faces,user_values,out_intersections,out_distances,out_faces);

    // Recopie en sortie les valeurs dans les variables correspondantes.
    ENUMERATE_ITEM(iitem,ray_family->allItems()){
      Integer index = iitem.index();
      intersections[iitem] = out_intersections[index];
      distances[iitem] = out_distances[index];
      rays_face[iitem] = out_faces[index];
    }

  }

  // Pour test, écrit les rayons et leur point d'impact.
  {
    Integer nb_new_ray = ray_family->nbItem();
    RealUniqueArray local_distances(nb_new_ray);
    local_directions.resize(nb_new_ray);
    local_positions.resize(nb_new_ray);
    ENUMERATE_ITEM(iitem,ray_family->allItems()){
      Integer index = iitem.index();
      local_distances[index] = distances[iitem];
      local_directions[index] = rays_direction[iitem];
      local_positions[index] = rays_position[iitem];
    }
    _writeSegments(my_rank,local_positions,local_directions,local_distances);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicRayMeshIntersection::
_writeSegments(Int32 rank,
               Real3ConstArrayView positions,
               Real3ConstArrayView directions,
               RealConstArrayView distances)
{
  Lima::Maillage lima("segments");
  lima.dimension(Lima::D3);

  Integer nb_segment = positions.size();
  UniqueArray<Lima::Noeud> lm_nodes(nb_segment*2);
  for( Integer i=0; i<nb_segment; ++i ){
    Real t = distances[i];
    if (t<0.0)
      t = 10.0;
    lm_nodes[i*2].set_x(positions[i].x);
    lm_nodes[i*2].set_y(positions[i].y);
    lm_nodes[i*2].set_z(positions[i].z);
    
    lm_nodes[(i*2)+1].set_x(positions[i].x + t*directions[i].x);
    lm_nodes[(i*2)+1].set_y(positions[i].y + t*directions[i].y);
    lm_nodes[(i*2)+1].set_z(positions[i].z + t*directions[i].z);
    lima.ajouter(lm_nodes[(i*2)]);
    lima.ajouter(lm_nodes[(i*2)+1]);
		lima.ajouter(Lima::Bras(lm_nodes[i*2],lm_nodes[(i*2)+1]));
  }
  StringBuilder sb("segments");
  sb+=rank;
  sb+=".unf";
  std::string s(sb.toString().localstr());
  lima.ecrire(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(BasicRayMeshIntersection,IRayMeshIntersection,
																	 BasicRayMeshIntersection);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
