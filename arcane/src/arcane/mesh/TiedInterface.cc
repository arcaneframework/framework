// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterface.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Informations sur les semi-conformitées du maillage.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/mesh/TiedInterface.h"

#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemGroup.h"
#include "arcane/Item.h"
#include "arcane/ISubDomain.h"
#include "arcane/VariableTypes.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemCompare.h"
#include "arcane/IParallelMng.h"
#include "arcane/GeometricUtilities.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMeshUtilities.h"

#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceBuilderInfos
{
 public:
  void printInfos()
  {
    std::ostream& o = cout;
    o << " MasterFacesUid=" << m_master_faces_uid.size()
      << " SlaveNodesUid=" << m_slave_nodes_uid.size()
      << " SlaveFacesUid=" << m_slave_faces_uid.size()
      << " MasterFacesNbSlaveNode=" << m_master_faces_nb_slave_node.size()
      << " MasterFacesNbSlaveFace=" << m_master_faces_nb_slave_face.size()
      << " SlaveNodesIso=" << m_slave_nodes_iso.size()
      << " MasterFacesSlaveFaceIndex=" << m_master_faces_slave_face_index.size()
      << " MasterFacesSlaveNodeIndex=" << m_master_faces_slave_node_index.size()
      << "\n";
  };
 public:
  //! Liste des uniqueId() des faces maitres
  UniqueArray<ItemUniqueId> m_master_faces_uid;
  //! Liste des uniqueId() des noeuds esclaves
  UniqueArray<ItemUniqueId> m_slave_nodes_uid;
  //! Liste des uniqueId() des faces esclaves
  UniqueArray<ItemUniqueId> m_slave_faces_uid;
  //! Nombre de noeuds esclaves pour chaque face maitre
  IntegerUniqueArray m_master_faces_nb_slave_node;
  //! Nombre de faces esclaves pour chaque face maitre
  IntegerUniqueArray m_master_faces_nb_slave_face;
  //! Liste des coordonnées iso-barycentriques des noeuds esclaves
  Real2UniqueArray m_slave_nodes_iso;

  IntegerUniqueArray m_master_faces_slave_face_index;
  IntegerUniqueArray m_master_faces_slave_node_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceFaceInfoMng
{
 public:
  Integer size() const { return m_nodes_unique_id.size(); }
  void add(ItemUniqueId node_unique_id)
  {
    m_nodes_unique_id.add(node_unique_id);
  }
public:
  UniqueArray<ItemUniqueId> m_nodes_unique_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceNodeInfo
{
 public:
  TiedInterfaceNodeInfo()
  : m_unique_id(NULL_ITEM_ID) {}
  TiedInterfaceNodeInfo(ItemUniqueId node_uid)
  : m_unique_id(node_uid) {}
 public:
  ItemUniqueId uniqueId() const
  {
    return m_unique_id;
  }
  void addConnectedFace(ItemUniqueId uid)
  {
    for( Integer i=0,is=m_connected_master_faces.size(); i<is; ++i )
      if (m_connected_master_faces[i]==uid)
        return;
    m_connected_master_faces.add(uid);
  }

 private:
  //! Numéro de ce noeud
  ItemUniqueId m_unique_id;
 public:

 public:
  //! Coordonnées de ce noeud
  Real3 m_coord;

  //! Liste des uniqueId() des faces maîtres auquel ce noeud peut être connecté
  SharedArray<ItemUniqueId> m_connected_master_faces;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Face maitre ou esclave d'une interface.
 *
 * Cet objet peut être copié mais ne doit pas être conservé une
 * fois le gestionnnaire \a m_mng associé détruit.
 */
class TiedInterfaceFace
{
 public:
  TiedInterfaceFace()
  : m_unique_id(NULL_ITEM_ID), m_cell_unique_id(NULL_ITEM_ID), m_owner(-1),
    m_nb_node(0), m_data_index(-1), m_mng(0) {}
  TiedInterfaceFace(ItemUniqueId unique_id,ItemUniqueId cell_unique_id,Integer nb_node,Integer owner,
                    Integer data_index,TiedInterfaceFaceInfoMng* mng)
  : m_unique_id(unique_id), m_cell_unique_id(cell_unique_id), m_owner(owner), m_nb_node(nb_node),
    m_data_index(data_index), m_mng(mng) {}
 public:
  ItemUniqueId uniqueId() const
  {
    return m_unique_id;
  }
  ItemUniqueId cellUniqueId() const
  {
    return m_cell_unique_id;
  }
  Integer nbNode() const
  {
    return m_nb_node;
  }
  ItemUniqueId nodeUniqueId(Integer i) const
  {
    return m_mng->m_nodes_unique_id[ m_data_index + i ];
  }
  Integer owner() const
  {
    return m_owner;
  }
  void setCenter(Real3 center)
  {
    m_center = center;
  }
  Real3 center() const
  {
    return m_center;
  }
 private:
  ItemUniqueId m_unique_id;
  ItemUniqueId m_cell_unique_id;
  Integer m_owner;
  Integer m_nb_node;
  Real3 m_center;
 public:
  Integer m_data_index;
  TiedInterfaceFaceInfoMng* m_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceMasterFace
{
 public:
  class NodeInfo
  {
  public:
    NodeInfo() : m_unique_id(NULL_ITEM_ID), m_alpha(-1.0), m_beta(-1.0) {}
    NodeInfo(ItemUniqueId uid)
    : m_unique_id(uid), m_alpha(-1.0), m_beta(-1.0) {}
  public:
    ItemUniqueId m_unique_id;
    Real m_alpha;
    Real m_beta;
  };
  typedef HashTableMapT<ItemUniqueId,NodeInfo> NodeInfoList;
 public:
  TiedInterfaceMasterFace()
  : m_unique_id(NULL_ITEM_ID) {}
  TiedInterfaceMasterFace(ItemUniqueId unique_id)
  : m_unique_id(unique_id) {}
 public:
  ItemUniqueId uniqueId() const
  { return m_unique_id; }
 public:
  ItemUniqueId m_unique_id;
  SharedArray<TiedInterfaceFace*> m_slave_faces;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceSortedNodeInfo
{
 public:
  TiedInterfaceSortedNodeInfo(ItemUniqueId uid,Real alpha,Real beta)
  : m_uid(uid), m_alpha(alpha), m_beta(beta){ }
 public:
  bool operator<(const TiedInterfaceSortedNodeInfo& rhs) const
  {
    if (m_alpha!=rhs.m_alpha)
      return m_alpha<rhs.m_alpha;
    if (m_beta!=rhs.m_beta)
      return (m_beta<rhs.m_beta);
    return (m_uid<rhs.m_uid);
  }
 public:
  ItemUniqueId uniqueId() const { return m_uid; }
 public:
  ItemUniqueId m_uid;
  Real m_alpha;
  Real m_beta;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceStructurationInfo
{
 public:
  TiedInterfaceStructurationInfo(ItemUniqueId node_uid,Real alpha,Real beta)
  : m_node_uid(node_uid), m_alpha(alpha), m_beta(beta), m_x(-1), m_y(-1) 
  {
  }
  TiedInterfaceStructurationInfo()
  : m_node_uid(NULL_ITEM_UNIQUE_ID), m_alpha(0.0), m_beta(0.0), m_x(-1), m_y(-1) 
  {
  }
 public:
  bool operator<(const TiedInterfaceStructurationInfo& rhs) const
  {
    if (m_x!=rhs.m_x)
      return m_x<rhs.m_x;
    if (m_y!=rhs.m_y)
      return (m_y<rhs.m_y);
    return false;
  }
  public:
  void setStructuration(Integer x,Integer y)
  {
    if (m_x!=(-1) && x!=m_x)
      ARCANE_FATAL("already set with a different x value old={0} new={1}",m_x,x);
    if (m_y!=(-1) && y!=m_y)
      ARCANE_FATAL("already set with a different y value old={0} new={1}",m_y,y);
    m_x = x;
    m_y = y;
  }
  bool hasStructuration() const
  {
    return (m_x!=(-1) && m_y!=(-1));
  }
 public:
  ItemUniqueId m_node_uid;
  Real m_alpha;
  Real m_beta;
  Integer m_x; //!< Structuration X
  Integer m_y; //!< Structuration Y
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des informations d'une interface semi-conforme.
 *
 * L'algorithme actuel présente les limitations suivantes:
 * - il ne s'applique qu'aux surfaces composées uniquement de quadrangles en 3D
 * et aux arêtes en 2D.
 *
 * - une interface semi-conforme est composée de deux surfaces, l'une
 * appelée esclave et l'autre maitre. La surface esclave contient
 * l'ensemble des faces qui sont le plus maillées. Cette surface
 * est spécifiée par l'utilisateur. La surface maitre est celle
 * contenant les faces les plus grossièrement maillées.
 *
 * Le but est de calculer l'ensemble des faces maitres et esclaves, pour
 * chaque face maitre la liste de ses faces esclaves et pour chaque
 * noeud de face esclave ses coordonnées iso-barycentrique dans
 * la face maitre correspondante.
 *
 * Le fonctionnement de l'algorithme est le suivant:
 * - on connait l'ensemble des faces esclaves. Il faut déterminer l'ensemble
 * des faces maitres. Pour cela, on parcours toutes les faces esclaves
 * et on marque leurs noeuds. Il suffit ensuite de parcourir toutes les
 * faces externes du maillage et si une face a tous ses noeuds marqués
 * et qu'elle n'est pas esclave, il s'agit d'une face maitre.
 * - TODO CONTINUER.
 */
class TiedInterfaceBuilder
: public TraceAccessor
{
 public:
  typedef HashTableMapT<ItemUniqueId,TiedInterfaceNodeInfo> NodeInfoList;
  typedef HashTableMapT<ItemUniqueId,TiedInterfaceFace> TiedInterfaceFaceMap;
  typedef HashTableMapEnumeratorT<ItemUniqueId,TiedInterfaceNodeInfo> NodeInfoListEnumerator;
  typedef HashTableMapEnumeratorT<ItemUniqueId,TiedInterfaceFace> TiedInterfaceFaceMapEnumerator;
  typedef std::set<TiedInterfaceSortedNodeInfo> SortedNodeInfoSet;
  typedef HashTableMapT<ItemUniqueId,TiedInterfaceStructurationInfo> StructurationMap;
  typedef HashTableMapEnumeratorT<ItemUniqueId,TiedInterfaceStructurationInfo> StructurationMapEnumerator;
 public:
  TiedInterfaceBuilder(IMesh* mesh,const FaceGroup& slave_interface,bool use_own,bool is_debug);
  void setPlanarTolerance(Real tolerance);
 public:
  void computeInterfaceConnections(bool allow_communication);
  void computeInterfaceInfos(TiedInterfaceBuilderInfos& infos,bool is_structured);
  void changeOwners(Int64Array& linked_cells,Int32Array& linked_owers);
  void changeOwnersOld();
  const FaceGroup& masterInterface() const { return m_master_interface; }
 private:
  bool m_is_debug;
  IMesh* m_mesh;
  VariableNodeReal3 m_nodes_coord;
  TiedInterfaceFaceInfoMng m_face_info_mng;
  //UniqueArray<TiedInterfaceFace> m_slave_faces;
  NodeInfoList m_nodes_info;
  TiedInterfaceFaceMap m_slave_faces;
  TiedInterfaceFaceMap m_master_faces;
  String m_slave_interface_name;
  FaceGroup m_slave_interface;
  FaceGroup m_master_interface;
  //! Table indiquant pour chaque face esclave, le uid de la face maitre correspondante.
  HashTableMapT<ItemUniqueId,ItemUniqueId> m_slave_faces_master_face_uid;
  Real m_planar_tolerance;

 private:
  GeometricUtilities::ProjectionInfo _findProjection(const TiedInterfaceFace& face,Real3 point);
  void _searchMasterFaces(Array<ItemUniqueId>& slave_faces_to_process,
                          Array<ItemUniqueId>& remaining_slave_faces);
  bool _isInsideFace(const TiedInterfaceFace& face,Real3 point);
  Real3 _computeNormale(const TiedInterfaceFace& face);
  void _computeMasterInterface();
  void _gatherFaces(ConstArrayView<ItemUniqueId> faces_to_send,
                    TiedInterfaceFaceMap& face_map);
  void _gatherAllNodesInfo();
  void _printFaces(std::ostream& o,TiedInterfaceFaceMap& face_map);
  void _computeProjectionInfos(TiedInterfaceBuilderInfos& infos,bool is_structured);
  void _addFaceToList(const Face& face,TiedInterfaceFaceMap& face_map);
  void _detectStructuration(const TiedInterfaceMasterFace& master_face,
                            StructurationMap& nodes);
  void _detectStructurationRecursive(Array<ItemUniqueId>& slave_faces_to_process,
                                     Array<ItemUniqueId>& remaining_slave_faces,
                                     StructurationMap& slave_nodes);
  void _removeMasterFacesWithNoSlave();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceBuilder::
TiedInterfaceBuilder(IMesh* mesh,const FaceGroup& slave_interface,
                     bool use_own,bool is_debug)
: TraceAccessor(mesh->traceMng())
, m_is_debug(is_debug)
, m_mesh(mesh)
, m_nodes_coord(m_mesh->toPrimaryMesh()->nodesCoordinates())
, m_nodes_info(1000,true)
, m_slave_faces(5000,true)
, m_master_faces(1000,true)
, m_slave_interface_name(slave_interface.name())
, m_slave_interface(slave_interface)
, m_slave_faces_master_face_uid(1000,true)
, m_planar_tolerance(0.0)
{
  if (use_own)
    m_slave_interface = slave_interface.own();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
setPlanarTolerance(Real tolerance)
{
  m_planar_tolerance = tolerance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_searchMasterFaces(Array<ItemUniqueId>& slave_faces_to_process,
                   Array<ItemUniqueId>& remaining_slave_faces)
{
  String func_name = "TiedInterfaceBuilder::_searchMasterFaces()";

  std::set<ItemUniqueId> master_faces_to_test;
  UniqueArray<ItemUniqueId> master_faces_found;
  master_faces_found.reserve(100);
  Real3UniqueArray slave_face_nodes_coord;
  slave_face_nodes_coord.reserve(100);
  Real3UniqueArray triangle_centers;
  triangle_centers.reserve(100);

  if (m_is_debug){
    info() << "nb slave faces to process=" << slave_faces_to_process.size();
    //info() << "nb remaining slave faces=" << remaining_slave_faces.size();
  }
  for( Integer islaveface=0,  izz=slave_faces_to_process.size(); islaveface<izz; ++islaveface ){
    master_faces_to_test.clear();
    master_faces_found.clear();

    TiedInterfaceFace& face = m_slave_faces[ slave_faces_to_process[islaveface] ];
    
    slave_face_nodes_coord.clear();

    Integer nb_node = face.nbNode();
    triangle_centers.resize(nb_node);
    
    for( Integer inode=0; inode<nb_node; ++inode ){
      TiedInterfaceNodeInfo& node_info = m_nodes_info[face.nodeUniqueId(inode)];
      slave_face_nodes_coord.add(node_info.m_coord);
      for( Integer zface=0, zz=node_info.m_connected_master_faces.size(); zface<zz; ++zface ){
        master_faces_to_test.insert(node_info.m_connected_master_faces[zface]);
      }
    }

    for( Integer inode=0; inode<nb_node; ++inode ){
      triangle_centers[inode] = (slave_face_nodes_coord[inode]
                                 + slave_face_nodes_coord[(inode+1)%nb_node]
                                 + face.center() ) / 3.;
    }

    // Parcours la liste des faces maitres possibles, et regarde si l'un des barycentres
    // de la decomposition en triangle de la face esclave est à l'intérieur d'une de ces faces.
    for( std::set<ItemUniqueId>::const_iterator i_master_face(master_faces_to_test.begin());
         i_master_face!=master_faces_to_test.end(); ++i_master_face ){
      TiedInterfaceFaceMap::Data* data = m_master_faces.lookup(*i_master_face);
      if (!data)
        ARCANE_FATAL("INTERNAL: Can not find face uid={0}",*i_master_face);
      const TiedInterfaceFace& master_face = data->value();
      bool is_found = false;
      for( Integer inode=0; inode<nb_node; ++inode ){
        if (_isInsideFace(master_face,triangle_centers[inode])){
//           info() << "Master found " << master_face.uniqueId() << " vs slave:" << face.uniqueId();
//           info() << "Master coords:" << m_nodes_info[master_face.nodeUniqueId(0)].m_coord
//                  << " " << m_nodes_info[master_face.nodeUniqueId(1)].m_coord
//                  << " " << m_nodes_info[master_face.nodeUniqueId(2)].m_coord
//                  << " " << m_nodes_info[master_face.nodeUniqueId(3)].m_coord;
//           info() << "Slave coords:" << m_nodes_info[face.nodeUniqueId(0)].m_coord
//                  << " " << m_nodes_info[face.nodeUniqueId(1)].m_coord
//                  << " " << m_nodes_info[face.nodeUniqueId(2)].m_coord
//                  << " " << m_nodes_info[face.nodeUniqueId(3)].m_coord;
//           info() << "This point : " << inode << " " << triangle_centers[inode];
          is_found = true;
          break;
        }
      }
       
      if (is_found){
        //info() << "SlaveFace " << face.uniqueId() << " found in face "
        //<< master_face.uniqueId() << " d=" << best_d.m_distance;
        master_faces_found.add(master_face.uniqueId());
      }
    }
    switch (master_faces_found.size()){
    case 1:
      {
        ItemUniqueId master_face_uid = master_faces_found[0];
        //info() << "** GOOD: SlaveFace " << face.uniqueId() << " found in face " << master_face_uid;
        for( Integer inode=0; inode<nb_node; ++inode ){
          TiedInterfaceNodeInfo& node_info = m_nodes_info[face.nodeUniqueId(inode)];
          node_info.m_connected_master_faces.add(master_face_uid);
        }
        //face.setMasterFace(master_face_uid);
        m_slave_faces_master_face_uid.add(face.uniqueId(),master_face_uid);
      }
      break;
    case 0:
      // Pas trouvé, ajoute dans la liste pour le test suivant.
      //info() << "** BAD: SlaveFace " << face.uniqueId();
      remaining_slave_faces.add(face.uniqueId());
      break;
    default:
      {
        // Ce cas peut se présenter si en 2D ou en 3D une maille possède une soudure sur
        // deux faces connectées. Dans ce cas, il y a souvent plusieurs
        // faces maitres possibles. Pour les discréminer, on choisit
        // la face maitre dont l'orientation (la normale) est la plus colineaire
        // avec la maille esclave. Pour cela, il suffit de calculer
        // le produit scalaire entre la face esclave et chaque face maitre
        // potentielle et de prendre le plus grand en valeur absolue (pour eviter
        // les problemes d'orientation).
        Real3 face_normale = _computeNormale(face);
        OStringStream ostr;
        ostr() << "Too many master faces for a slave face (max=1) "
               << " nb_master=" << master_faces_found.size()
               << " (slave_face=" << face.uniqueId() << ",normal=" << face_normale << ")"
               << " master_list=";
        Real max_dot = -1.0;
        Integer keep_index = -1;
        for( Integer zz=0; zz< master_faces_found.size(); ++zz ){
          ItemUniqueId muid = master_faces_found[zz];
          const TiedInterfaceFace& master_face = m_master_faces.lookupValue(muid);
          Real3 n = _computeNormale(master_face);
          Real d = math::scaMul(n,face_normale);
          if (math::abs(d)>max_dot){
            max_dot = math::abs(d);
            keep_index = zz;
          }
          ostr() << " (uid=" << muid << ",nb_node=" << master_face.nbNode()
                 << " center=" << master_face.center()
                 << " normale=" << n
                 << " dot=" << d
                 << ")";
        }
        if (keep_index>=0){
          info() << func_name << " " << ostr.str() << ". Keeping index=" << keep_index;
          {
            ItemUniqueId master_face_uid = master_faces_found[keep_index];
            for( Integer inode=0; inode<nb_node; ++inode ){
              TiedInterfaceNodeInfo& node_info = m_nodes_info[face.nodeUniqueId(inode)];
              node_info.m_connected_master_faces.add(master_face_uid);
            }
            //face.setMasterFace(master_face_uid);
            m_slave_faces_master_face_uid.add(face.uniqueId(),master_face_uid);
          }
        }
        else
          ARCANE_FATAL(ostr.str());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule la projection d'un point sur une face.
 *
 * En 2D, la face est une arête et la projection est simple.
 *
 * En 3D, si la face est un triangle, la projection est aussi simple
 * puisqu'il s'agit de la projection sur un plan.
 * Pour une face comportant plus de 3 noeuds, ses noeuds ne sont
 * pas nécessairement coplanaires. On décompose alors la face en
 * triangles dont un des sommets est le barycentre de la face et on
 * calcule la projection du point \a point sur chacun de ces triangles.
 * On a donc autant de projetés que de triangles. On conserve
 * celui qui est à l'intérieur d'un de ces triangles.
 * Il peut arriver pour des raisons liées au calcul numérique que
 * le point soit bien à l'intérieur de la face mais dans aucun
 * de ses triangles (par exemple s'il est sur une diagonale).
 * Dans ce cas, on prend comme projeté celui qui est le plus
 * proche du point \a point.
 */
GeometricUtilities::ProjectionInfo TiedInterfaceBuilder::
_findProjection(const TiedInterfaceFace& face,Real3 point)
{
  Integer nb_node = face.nbNode();

  GeometricUtilities::ProjectionInfo min_distance;

  if (nb_node>3){
    for( Integer inode=0; inode<nb_node; ++inode ){
      Real3 v1 = m_nodes_info[face.nodeUniqueId(inode)].m_coord;
      Real3 v2 = m_nodes_info[face.nodeUniqueId((inode+1)%nb_node)].m_coord;
      GeometricUtilities::ProjectionInfo d = min_distance.projection(face.center(),v1,v2,point);
      //info() << " DISTANCE: r=" << region << " d=" << d;
      if (d.m_region==0){
        if (min_distance.m_region!=0){
          min_distance = d;
        }
        else{
          if (d.m_distance<min_distance.m_distance){
            min_distance = d;
          }
        }
      }
      else{
        if (min_distance.m_region!=0){
          if (d.m_distance<min_distance.m_distance){
            min_distance = d;
          }
        }
      }
    }
  }
  else if (nb_node==3){
    Real3 v1 = m_nodes_info[face.nodeUniqueId(0)].m_coord;
    Real3 v2 = m_nodes_info[face.nodeUniqueId(1)].m_coord;
    Real3 v3 = m_nodes_info[face.nodeUniqueId(2)].m_coord;
    min_distance = min_distance.projection(v1,v2,v3,point);
  }
  else if (nb_node==2){
    Real3 v1 = m_nodes_info[face.nodeUniqueId(0)].m_coord;
    Real3 v2 = m_nodes_info[face.nodeUniqueId(1)].m_coord;
    min_distance = min_distance.projection(v1,v2,point);
  }
  return min_distance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 TiedInterfaceBuilder::
_computeNormale(const TiedInterfaceFace& face)
{
  Integer nb_node = face.nbNode();
  Real3 normale;
  if (nb_node>=3){
    for( Integer inode=0; inode<nb_node; ++inode ){
      Real3 v1 = m_nodes_info[face.nodeUniqueId(inode)].m_coord;
      Real3 v2 = m_nodes_info[face.nodeUniqueId((inode+1)%nb_node)].m_coord;
      Real3 v3 = m_nodes_info[face.nodeUniqueId((inode+2)%nb_node)].m_coord;
      Real3 vd_a = v1 - v2;
      Real3 vd_b = v3 - v2;
      Real3 local_normale = math::vecMul(vd_a,vd_b);
      normale += local_normale;
    }
  }
  else if (nb_node==2){
    Real3 v1 = m_nodes_info[face.nodeUniqueId(0)].m_coord;
    Real3 v2 = m_nodes_info[face.nodeUniqueId(1)].m_coord;
    if (v1.z!=0.0 || v2.z!=0.0)
      throw NotImplementedException(A_FUNCINFO,"edge in 3D space");
    Real x = v2.x - v1.x;
    Real y = v2.y - v1.y;
    normale = Real3(-y,x,0.0);
  }
  else
    throw NotSupportedException(A_FUNCINFO,"can not compute normal of face with 0 or 1 node");
  return normale.normalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TiedInterfaceBuilder::
_isInsideFace(const TiedInterfaceFace& face,Real3 point)
{
  Integer nb_node = face.nbNode();
  if (nb_node>2){
    for( Integer inode=0; inode<nb_node; ++inode ){
      Real3 v1 = m_nodes_info[face.nodeUniqueId(inode)].m_coord;
      Real3 v2 = m_nodes_info[face.nodeUniqueId((inode+1)%nb_node)].m_coord;
      if (m_planar_tolerance!=0.0) { // Ne fait le calcul que si le test est utile
        Real ecart = math::mixteMul(point-face.center(),math::normalizeReal3(v1-face.center()),math::normalizeReal3(v2-face.center()));
        if (math::abs(ecart) > m_planar_tolerance * math::normeR3(v1-v2)) {
          if (m_is_debug)
            info() << "Reject non planar projection " << point << " from face uid=" << face.uniqueId();
          return false;
        }
      }
      if (GeometricUtilities::ProjectionInfo::isInside(face.center(),v1,v2,point))
        return true;
    }
  }
  else if (nb_node==2){
    Real3 v1 = m_nodes_info[face.nodeUniqueId(0)].m_coord;
    Real3 v2 = m_nodes_info[face.nodeUniqueId(1)].m_coord;
    if (GeometricUtilities::ProjectionInfo::isInside(v1,v2,point))
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine la surface maitre de l'interface.
 *
 * En considérant que le maillage est semi-conforme, cela signifie que
 * tous les noeuds des faces maîtres de l'interface appartiennent à une
 * face esclave. Il suffit donc pour déterminer ces faces maîtres de
 * parcourir la liste des faces esclaves, de marquer les noeuds les
 * noeuds de ces faces. Une face est alors considérée comme maître si
 * l'ensemble de ses noeuds est marqué.
 *
 * \note Cet algorithme peut potentiellement retourner plus de faces
 * maîtres qu'il y en a réellement dans le cas où une maille à des soudures
 * sur plusieurs côtés. Cela n'est pas très grave car aucune face
 * esclave ne sera trouvé pour ces faces maîtres et on les supprimera
 * du groupe de faces maitres.
 */
void TiedInterfaceBuilder::
_computeMasterInterface()
{
  // Marqueur faces esclaves
  std::set<ItemUniqueId> slave_faces_flag;
  
  Integer nb_slave_face = m_slave_interface.size();
  m_nodes_info.resize((nb_slave_face*2)+5);
  m_master_faces.resize(nb_slave_face+5);
  m_slave_faces.resize((nb_slave_face*2)+5);

  // Construit les informations nécessaires concernant les
  // faces esclaves et marque l'ensemble des noeuds.
  ENUMERATE_FACE(iface,m_slave_interface){
    const Face& face = *iface;
    slave_faces_flag.insert(face.uniqueId());
    _addFaceToList(face,m_slave_faces);
  }

  info() << "SLAVE_INTERFACE: nb_face=" << m_slave_interface.size();
  // A partir des noeuds marqués, détermine les faces maîtres
  Int32UniqueArray master_faces_lid;
  bool has_not_handled_face = false;
  ENUMERATE_FACE(iface,m_mesh->outerFaces()){
    const Face& face = *iface;
    // Vérifie qu'il ne s'agit pas d'une face esclave
    if (slave_faces_flag.find(face.uniqueId())!=slave_faces_flag.end())
      continue;
    Integer nb_node = face.nbNode();
    bool is_master_face = true;
    // Une face est maître si chacun de ses noeuds est dans la liste des noeuds esclaves
    for( Node node : face.nodes() ){
      ItemUniqueId uid = node.uniqueId();
      if (!m_nodes_info.lookup(uid)){
        is_master_face = false;
        break;
      }
    }
    if (is_master_face){
      // Pour l'instant, supporte uniquement les faces à 2 et 4 noeuds.
      if (nb_node!=4 && nb_node!=2)
        has_not_handled_face = true;
      master_faces_lid.add(face.localId());
      for( Node node : face.nodes() ){
        TiedInterfaceNodeInfo& znode = m_nodes_info[node.uniqueId()];
        znode.m_connected_master_faces.add(face.uniqueId());
      }
      _addFaceToList(face,m_master_faces);
    }
  }
  if (has_not_handled_face)
    ARCANE_FATAL("Some faces of the tied interface '{0}' has incorrect number of nodes (should be 2 or 4)",
                 m_slave_interface.name());

  m_master_interface = m_mesh->faceFamily()->createGroup(m_slave_interface_name+"_MASTER",
                                                         master_faces_lid,true);
  info() << "MASTER_INTERFACE: nb_face=" << m_master_interface.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_addFaceToList(const Face& face,TiedInterfaceFaceMap& face_map)
{
  Integer nb_node = face.nbNode();
  Real3 center(0.,0.,0.);
  Integer data_index = m_face_info_mng.size();
  for( Node node : face.nodes() ){
    Real3 node_coord = m_nodes_coord[node];
    ItemUniqueId uid = node.uniqueId();
    NodeInfoList::Data* i = m_nodes_info.lookup(uid);
    if (!i){
      TiedInterfaceNodeInfo node_info(uid);
      node_info.m_coord = node_coord;
      m_nodes_info.add(uid,node_info);
    }
    m_face_info_mng.add(uid);
    center += node_coord;
  }
  center /= nb_node;
  //info() << "ADD FACE uid=" << face.uniqueId() << " nb_node="
  //<< nb_node << " center=" << center;
  TiedInterfaceFace sf(face.uniqueId(),face.cell(0).uniqueId(),nb_node,face.owner(),data_index,&m_face_info_mng);
  sf.setCenter(center);
  face_map.add(face.uniqueId(),sf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Supprime du groupe des faces maîtres les faces qui ne sont
 * connectées à aucune face esclave.
 */
void TiedInterfaceBuilder::
_removeMasterFacesWithNoSlave()
{
  std::set<ItemUniqueId> master_face_with_slave;

  for( TiedInterfaceFaceMapEnumerator i(m_slave_faces); ++i; ){
    TiedInterfaceFace& slave_face = *i;
    ItemUniqueId master_uid = m_slave_faces_master_face_uid[slave_face.uniqueId()];
    master_face_with_slave.insert(master_uid);
  }

  Int32UniqueArray local_ids_to_remove;
  ENUMERATE_FACE(iface,m_master_interface.own()){
    Face face = *iface;
    ItemUniqueId uid = face.uniqueId();
    if (master_face_with_slave.find(uid)==master_face_with_slave.end()){
      local_ids_to_remove.add(iface.itemLocalId());
    }
  }

  if (!local_ids_to_remove.empty()){
    info() << "Removing faces from master list name=" << m_master_interface.name()
           << " ids=" << local_ids_to_remove;
    m_master_interface.removeItems(local_ids_to_remove,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_computeProjectionInfos(TiedInterfaceBuilderInfos& infos,bool is_structured)
{
  typedef HashTableMapT<ItemUniqueId,TiedInterfaceMasterFace> MasterFaceList;
  Integer nb_master_face = m_master_interface.size();
  MasterFaceList m_master_faces_full((nb_master_face*2) + 1,true);

  infos.m_master_faces_uid.reserve(nb_master_face);
  infos.m_master_faces_nb_slave_node.reserve(nb_master_face);
  infos.m_master_faces_nb_slave_face.reserve(nb_master_face);

  TiedInterfaceMasterFace null_value;
  ENUMERATE_FACE(iface,m_master_interface.own()){
    const Face& face = *iface;
    ItemUniqueId master_uid = face.uniqueId();
    m_master_faces_full.add(master_uid,TiedInterfaceMasterFace(master_uid));
  }
  
  for( TiedInterfaceFaceMapEnumerator i(m_slave_faces); ++i; ){
    TiedInterfaceFace& slave_face = *i;
    ItemUniqueId master_uid = m_slave_faces_master_face_uid[slave_face.uniqueId()];
    TiedInterfaceMasterFace& mf = m_master_faces_full[master_uid];
    mf.m_slave_faces.add(&slave_face);
  }

  ENUMERATE_FACE(iface,m_master_interface.own()){
    Face face = *iface;
    ItemUniqueId master_uid = face.uniqueId();
    //TiedInterfaceMasterFace& mf = *enumerator;
    TiedInterfaceMasterFace& mf = m_master_faces_full[master_uid];
    Integer master_face_nb_slave_face = mf.m_slave_faces.size();
    //Int64 master_uid = mf.uniqueId();
    if (m_is_debug)
      info() << "MASTER FACE uid=" << master_uid << " NB_SLAVE_FACE=" << master_face_nb_slave_face
             << " face_owner=" << face.owner();
    infos.m_master_faces_uid.add(master_uid);
    infos.m_master_faces_nb_slave_face.add(master_face_nb_slave_face);
    infos.m_master_faces_slave_face_index.add(infos.m_slave_faces_uid.size());
    for( Integer zz=0; zz<master_face_nb_slave_face; ++zz ){
      ItemUniqueId slave_uid = mf.m_slave_faces[zz]->uniqueId();
      infos.m_slave_faces_uid.add(slave_uid);
    }
  }

  std::set<ItemUniqueId> slave_nodes_set;
  SortedNodeInfoSet slave_nodes_sorted_set;

  bool is_dimension_2d = m_mesh->dimension()==2;
  ENUMERATE_FACE(imasterface,m_master_interface.own()){
    Face master_face = *imasterface;
    ItemUniqueId master_uid = master_face.uniqueId();
    slave_nodes_set.clear();
    slave_nodes_sorted_set.clear();
    TiedInterfaceMasterFace& face2 = m_master_faces_full[master_uid];
    if (m_is_debug){
      Integer face2_nb_node = master_face.nbNode();
      info() << "MASTER FACE: uid=" << face2.uniqueId()
             << " cell_uid=" << master_face.cell(0).uniqueId()
             << " cell_owner=" << master_face.cell(0).owner()
             << " nb_node=" << face2_nb_node;
      for( Integer in2=0; in2<face2_nb_node; ++in2 )
        info() << "Node " << in2 << " uid=" << master_face.node(in2).uniqueId();
    }
    TiedInterfaceFace& face = m_master_faces[face2.uniqueId()];
    Integer nb_node = face.nbNode();
    Integer nb_slave_face = face2.m_slave_faces.size();
    for( Integer i=0; i<nb_slave_face; ++i ){
      const TiedInterfaceFace& slave_face = *face2.m_slave_faces[i];
      Integer slave_nb_node = slave_face.nbNode();
      for( Integer z=0; z<slave_nb_node; ++z ){
        ItemUniqueId node_uid = slave_face.nodeUniqueId(z);
        slave_nodes_set.insert(node_uid);
        //info() << "ADD NODE uid=" << node_uid;
      }
    }
    if (m_is_debug)
      info() << "MASTER FACE: NB_SLAVE_FACE=" << nb_slave_face
             << " NB_SLAVE_NODE=" << slave_nodes_set.size();
    infos.m_master_faces_slave_node_index.add(infos.m_slave_nodes_uid.size());
    infos.m_master_faces_nb_slave_node.add(CheckedConvert::toInteger(slave_nodes_set.size()));
    {
      std::set<ItemUniqueId>::const_iterator i_node = slave_nodes_set.begin();
      // Cas 3D, les faces sont des quadrangles
      if (nb_node==4){
        GeometricUtilities::QuadMapping face_mapping;
        for( Integer i=0; i<nb_node; ++i ){
          ItemUniqueId node_uid = face.nodeUniqueId(i);
          face_mapping.m_pos[i] = m_nodes_info[node_uid].m_coord;
        }

        for( ; i_node!=slave_nodes_set.end(); ++i_node ){
          ItemUniqueId node_uid = *i_node;
          TiedInterfaceNodeInfo& node_info = m_nodes_info[node_uid];
          Real3 uvw;
          Real3 point = node_info.m_coord;
          GeometricUtilities::ProjectionInfo projection = _findProjection(face,point);
          if (m_is_debug)
            info() << "POINT PROJECTION: uid=" << node_uid << ' '
                   << projection.m_projection << " r=" << projection.m_region
                   << " d=" << projection.m_distance
                   << " alpha=" << projection.m_alpha
                   << " beta=" << projection.m_beta;
          bool is_bad = face_mapping.cartesianToIso(point,uvw,0);
          if (m_is_debug)
            info() << "ISO1 =" << uvw;
          is_bad = face_mapping.cartesianToIso2(point,uvw,0);
          if (m_is_debug)
            info() << "ISO2 =" << uvw;
          if (is_bad){
            warning() << "Can not compute iso-coordinates for point " << point;
            face_mapping.cartesianToIso(point,uvw,traceMng());
          }

          if (math::abs(uvw.x)>1.1 || math::abs(uvw.y)>1.1 || math::abs(uvw.z)>1.1){
            info() << "BAD PROJECTION INFO";
            info() << "P0 = " << face_mapping.m_pos[0];
            info() << "P1 = " << face_mapping.m_pos[1];
            info() << "P2 = " << face_mapping.m_pos[2];
            info() << "P3 = " << face_mapping.m_pos[3];
            warning() << "Internal: bad iso value: " << uvw
                      << " node=" << node_uid << " pos=" << point
                      << " projection=" << projection.m_projection
                      << " face_uid=" << face.uniqueId()
                      << " cell_uid=" << face.cellUniqueId();
            face_mapping.cartesianToIso(point,uvw,traceMng());
            face_mapping.cartesianToIso(projection.m_projection,uvw,traceMng());
          }
          //else
          //info() << "POINT ISO1: " << uvw;
          //Real3 new_point = face_mapping.evaluatePosition(uvw);
          //info() << "POINT ISO_TO_CART: " << (new_point-point).abs();
          
          // Cas particulier des noeuds de la face esclave qui sont des noeuds maitres.
          // Dans ce cas, on fixe en dur leur coordonnées iso pour éviter une
          // perte de précision.
          if (node_uid==face.nodeUniqueId(0)){
            uvw.x = ARCANE_REAL(-1.0);
            uvw.y = ARCANE_REAL(-1.0);
          }
          else if (node_uid==face.nodeUniqueId(1)){
            uvw.x = ARCANE_REAL( 1.0);
            uvw.y = ARCANE_REAL(-1.0);
          }
          else if (node_uid==face.nodeUniqueId(2)){
            uvw.x = ARCANE_REAL(1.0);
            uvw.y = ARCANE_REAL(1.0);
          }
          else if (node_uid==face.nodeUniqueId(3)){
            uvw.x = ARCANE_REAL(-1.0);
            uvw.y = ARCANE_REAL( 1.0);
          }
          
          slave_nodes_sorted_set.insert(TiedInterfaceSortedNodeInfo(node_uid,uvw.x,uvw.y));
        }
      }
      // Case 2D, les faces sont des arêtes
      else if (nb_node==2){
        for( ; i_node!=slave_nodes_set.end(); ++i_node ){
          ItemUniqueId node_uid = *i_node;
          TiedInterfaceNodeInfo& node_info = m_nodes_info[node_uid];
          Real3 point = node_info.m_coord;
          GeometricUtilities::ProjectionInfo projection = _findProjection(face,point);
          //info() << "POINT PROJECTION: uid=" << node_uid << ' '
          //     << projection.m_projection
          //<< " r=" << projection.m_region << " d=" << projection.m_distance;
          // Cas particulier des noeuds de la face esclave qui sont des noeuds maitres.
          // Dans ce cas, on fixe en dur leur coordonnées iso pour éviter une
          // perte de précision.
          Real alpha = projection.m_alpha;
          if (node_uid==face.nodeUniqueId(0)){
            alpha = ARCANE_REAL(0.0);
          }
          else if (node_uid==face.nodeUniqueId(1)){
            alpha = ARCANE_REAL(1.0);
          }
          slave_nodes_sorted_set.insert(TiedInterfaceSortedNodeInfo(node_uid,alpha,0.0));
        }
      }
      else
        ARCANE_FATAL("Can not detect structuration for face with nb_node={0}."
                     " Valid values are 2 or 4",nb_node);
    }
    if (slave_nodes_sorted_set.size()!=slave_nodes_set.size()){
      ARCANE_FATAL("Internal: error sorting nodes in TiedInterface: bad compare");
    }
    // En cas de structuration, la détecte et recalcule
    // les coordonnées iso avec cette info.
    if (is_structured){
      StructurationMap struct_map(CheckedConvert::toInteger(slave_nodes_sorted_set.size()*2),true);
      SortedNodeInfoSet::const_iterator i_node = slave_nodes_sorted_set.begin();
      for( ; i_node!=slave_nodes_sorted_set.end(); ++i_node ){
        const TiedInterfaceSortedNodeInfo& node = *i_node;
        ItemUniqueId uid = node.uniqueId();
        TiedInterfaceStructurationInfo struct_info(uid,node.m_alpha,node.m_beta);
        struct_map.add(uid,struct_info);
        if (m_is_debug)
          info() << "Add to struct map node_uid=" << uid << " alpha=" << node.m_alpha << " beta=" << node.m_beta;
      }
      _detectStructuration(face2,struct_map);
      // Recherche la valeur de la structuration.
      // Comme on commence la structuration avec le noeud 0 de la face maitre,
      // C'est la valeur (x,y) du noeud 2 la maille maitre pour un quad, et
      // le noeud 1 pour une arête
      TiedInterfaceStructurationInfo sinfo = struct_map[face.nodeUniqueId(nb_node/2)];
      if (m_is_debug)
        info() << "********* STRUCTURE X=" << sinfo.m_x << " Y=" << sinfo.m_y;
      Integer sx = sinfo.m_x;
      Integer sy = sinfo.m_y;
      Real rx = (Real)sinfo.m_x;
      Real ry = (Real)sinfo.m_y;
      slave_nodes_sorted_set.clear();
      for( StructurationMapEnumerator imap(struct_map); ++imap; ){
        TiedInterfaceStructurationInfo& minfo = *imap;
        ItemUniqueId node_uid = minfo.m_node_uid;
        Real old_alpha = minfo.m_alpha;
        Real old_beta = minfo.m_beta;
        Real new_alpha = 0.0;
        Real new_beta = 0.0;
        if (is_dimension_2d){
          // En 2D, les coordonnées iso vont de 0.0 à 1.0
          if (sx!=0)
            new_alpha = (1.0 * minfo.m_x) / rx;
        }
        else{
          if (sx!=0)
            new_alpha = -1.0 + (2.0 * minfo.m_x) / rx;
          if (sy!=0)
            new_beta = -1.0 + (2.0 * minfo.m_y) / ry;
        }
        if (m_is_debug)
          info() << "NEW NODE oldx=" << old_alpha << " newx=" << new_alpha
                 << " oldy=" << old_beta << " newy=" << new_beta;
        slave_nodes_sorted_set.insert(TiedInterfaceSortedNodeInfo(node_uid,new_alpha,new_beta));
      }
    }
    {
      SortedNodeInfoSet::const_iterator i_node = slave_nodes_sorted_set.begin();
      for( ; i_node!=slave_nodes_sorted_set.end(); ++i_node ){
        const TiedInterfaceSortedNodeInfo& node = *i_node;
        if (m_is_debug)
          info() << "ADD TO SLAVE NODE: node_uid=" << node.m_uid
                 << " alpha=" << node.m_alpha
                 << " beta=" << node.m_beta;
        infos.m_slave_nodes_uid.add(node.m_uid);
        infos.m_slave_nodes_iso.add(Real2(node.m_alpha,node.m_beta));
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_detectStructuration(const TiedInterfaceMasterFace& master_face,
                     StructurationMap& slave_nodes)
{
  Integer nb_slave_face = master_face.m_slave_faces.size();
  // Positionne le premier noeud de la face comme étant le noeud (0,0)
  // de la structuration.
  TiedInterfaceFace& mface = m_master_faces[master_face.uniqueId()];
  TiedInterfaceStructurationInfo& sinfo = slave_nodes[mface.nodeUniqueId(0)];
  sinfo.setStructuration(0,0);
  //info() << "** SET STRUCTURATION node=" << mface.nodeUniqueId(0);
  UniqueArray<ItemUniqueId> slave_faces;
  UniqueArray<ItemUniqueId> remaining_slave_faces;
  for( Integer i=0; i<nb_slave_face; ++i ){
    const TiedInterfaceFace& slave_face = *master_face.m_slave_faces[i];
    slave_faces.add(slave_face.uniqueId());
  }

  for( Integer zz=0; zz<(nb_slave_face+1); ++zz ){
    remaining_slave_faces.clear();
    Integer nb_to_process = slave_faces.size();
    _detectStructurationRecursive(slave_faces,remaining_slave_faces,slave_nodes);
    Integer nb_remaining = remaining_slave_faces.size();
    if (nb_remaining==0)
      break;
    if (nb_to_process==nb_remaining){
      ARCANE_FATAL("Can not compute structuration for a tied interface"
                   " remaining_slaves={0}",nb_remaining);
    }
    slave_faces.copy(remaining_slave_faces);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_detectStructurationRecursive(Array<ItemUniqueId>& slave_faces,
                              Array<ItemUniqueId>& remaining_slave_faces,
                              StructurationMap& slave_nodes)
{
  // Parcours chaque face.
  // Une face peut être traitée si un de ses noeuds a déjà une structuration.
  // Si ce n'est pas le cas, la face est ajoutée à remaining_slave_faces
  // pour être traitée ultérieurement.
  // En cas de structuration, on prend le premier trouvé. A partir de ce noeud A,
  // on regarde le noeud suivant B dans la face. On regarde si ce
  // noeud B différe de notre noeud de référence suivant alpha ou beta.
  // S'il s'agit de alpha, il aura le même y et il sera en (x+1) si alpha
  // de B est supérieur à celui de A.
  // La même chose pour y si la différence est en beta.
  // NOTE. Comme la projection n'est pas forcément très bonne, deux
  // noeuds de même x par exemple peuvent avoir un alpha légèrement différent.
  // Pour déterminer si c'est alpha ou beta qui varie le plus, on
  // prend le plus grand de abs(A.alpha-B.alpha) et abs(A.beta-B.beta)
  for( Integer i=0, is=slave_faces.size(); i<is; ++i ){
    Integer node_index = (-1);
    TiedInterfaceStructurationInfo old_sinfo;
    const TiedInterfaceFace& face = m_slave_faces[ slave_faces[i] ];
    Integer nb_node = face.nbNode();
    for( Integer z=0; z<nb_node; ++z ){
      ItemUniqueId node_uid = face.nodeUniqueId(z);
      const TiedInterfaceStructurationInfo& sinfo = slave_nodes[node_uid];
      if (m_is_debug)
        info() << "CHECK NODE face_uid=" << face.uniqueId()
               << " node_uid=" << node_uid
               << " x=" << sinfo.m_x
               << " y=" << sinfo.m_y
               << " alpha=" << sinfo.m_alpha
               << " beta=" << sinfo.m_beta;
      if (sinfo.hasStructuration()){
        node_index = z;
        old_sinfo = sinfo;
        break;
      }
    }
    if (node_index==(-1)){
      remaining_slave_faces.add(slave_faces[i]);
      continue;
    }

    for( Integer z=1; z<nb_node; ++z ){
      ItemUniqueId next_uid = face.nodeUniqueId((node_index+z)%nb_node);
      TiedInterfaceStructurationInfo& next_info = slave_nodes[next_uid];
      Real diff_alpha = next_info.m_alpha - old_sinfo.m_alpha;
      Real diff_beta = next_info.m_beta - old_sinfo.m_beta;
      Integer x = old_sinfo.m_x;
      Integer y = old_sinfo.m_y;
      if (math::abs(diff_alpha)>math::abs(diff_beta)){
        // La variation est en x.
        if (next_info.m_alpha > old_sinfo.m_alpha){
          if (m_is_debug)
            info() << "SUP_ALPHA SET NEXT uid=" << next_uid << " x=" << (x+1) << " y=" << (y);
          next_info.setStructuration(x+1,y);
        }
        else{
          if (m_is_debug)
            info() << "INF_ALPHA SET NEXT uid=" << next_uid << " x=" << (x-1) << " y=" << (y);
          next_info.setStructuration(x-1,y);
        }
      }
      else{
        // La variation est en y.
        if (next_info.m_beta > old_sinfo.m_beta){
          if (m_is_debug)
            info() << "SUP_BETA  SET NEXT uid=" << next_uid << " x=" << (x) << " y=" << (y+1);
          next_info.setStructuration(x,y+1);
        }
        else{
          if (m_is_debug)
            info() << "INF_BETA  SET NEXT uid=" << next_uid << " x=" << (x) << " y=" << (y-1);
          next_info.setStructuration(x,y-1);
        }
      }
      old_sinfo = next_info;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
_printFaces(std::ostream& o,TiedInterfaceFaceMap& face_map)
{
  // Utilise une map pour trier les faces suivant leur uniqueId() afin
  // que l'affichage soit toujours le meme.
  typedef std::map<ItemUniqueId,TiedInterfaceFace*> FacesMap;

  FacesMap faces;
  for( TiedInterfaceFaceMapEnumerator i(face_map); ++i; ){
    TiedInterfaceFace* mf = &i.m_current_data->value();
    faces.insert(std::make_pair(mf->uniqueId(),mf));
  }
  
  for( FacesMap::const_iterator i(faces.begin()); i!=faces.end(); ++i ){
    TiedInterfaceFace& mf = *(i->second);
    Integer nb_node = mf.nbNode();
    o << " face=" << mf.uniqueId() << " nb_node=" << nb_node
      << " center=" << mf.center() << '\n';
    for( Integer z=0; z<nb_node; ++z ){
      ItemUniqueId nuid(mf.nodeUniqueId(z));
      o << " node uid=" << nuid << " coord=" << m_nodes_info[nuid].m_coord << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief envoie et récupère les informations sur les noeuds
 * de l'interface.
 *
 * Rassemble en parallèle les noeuds et la liste des faces maitres avec lesquelles
 * ils peuvent être connectectés.
 *
 * \warning Cette méthode ne doit pas être appelée en séquentiel.
 */
void TiedInterfaceBuilder::
_gatherAllNodesInfo()
{
  UniqueArray<TiedInterfaceNodeInfo*> nodes_info;
  Integer nb_to_send = 0;
  RealUniqueArray coords;
  Int64UniqueArray unique_ids;
  IntegerUniqueArray nb_connected_master_faces;
  Int64UniqueArray connected_master_faces;
  for( NodeInfoListEnumerator i(m_nodes_info); ++i; ){
    const TiedInterfaceNodeInfo& ni = *i;
    ++nb_to_send;
    unique_ids.add(ni.uniqueId());
    coords.add(ni.m_coord.x);
    coords.add(ni.m_coord.y);
    coords.add(ni.m_coord.z);
    Integer nb_connected = ni.m_connected_master_faces.size();
    nb_connected_master_faces.add(nb_connected);
    for( Integer z=0; z<nb_connected; ++z )
      connected_master_faces.add(ni.m_connected_master_faces[z]);
  }

  IParallelMng* pm = m_mesh->parallelMng();
  Integer nb_rank = pm->commSize();

  SerializeBuffer sbuf;
  sbuf.setMode(ISerializer::ModeReserve);
  sbuf.reserveInteger(1); // pour le nombre de noeuds
  sbuf.reserveInteger(1); // pour le nombre de faces connectées
  sbuf.reserve(DT_Int64,unique_ids.size()); // pour les uniqueId() des noeuds
  sbuf.reserveInteger(nb_connected_master_faces.size()); // pour le nombre de faces connectées
  sbuf.reserve(DT_Int64,connected_master_faces.size()); // pour les uniqueId() des faces
  sbuf.reserve(DT_Real,coords.size());

  sbuf.allocateBuffer();
  sbuf.setMode(ISerializer::ModePut);
  sbuf.putInteger(nb_to_send);
  sbuf.putInteger(connected_master_faces.size());
  sbuf.put(unique_ids);
  sbuf.put(nb_connected_master_faces);
  sbuf.put(connected_master_faces);
  sbuf.put(coords);

  SerializeBuffer recv_buf;
  pm->allGather(&sbuf,&recv_buf);
  recv_buf.setMode(ISerializer::ModeGet);

  for( Integer i=0; i<nb_rank; ++i ){
    Integer nb_node = recv_buf.getInteger();
    Integer nb_connected_face = recv_buf.getInteger();
    unique_ids.resize(nb_node);
    nb_connected_master_faces.resize(nb_node);
    connected_master_faces.resize(nb_connected_face);
    coords.resize(nb_node*3);

    recv_buf.get(unique_ids);
    recv_buf.get(nb_connected_master_faces);
    recv_buf.get(connected_master_faces);
    recv_buf.get(coords);

    // Parcours toutes les faces reçues si certaines sont absentes,
    // les ajoute.
    Integer face_index = 0;
    for( Integer z=0; z<nb_node; ++z ){
      Integer nb_face = nb_connected_master_faces[z];
      if (nb_face!=0){
        ItemUniqueId uid(unique_ids[z]);
        NodeInfoList::Data* data = m_nodes_info.lookup(uid);
        if (data){
          TiedInterfaceNodeInfo& ni = data->value();
          //info() << "RECEIVE NODE uid=" << uid << " nb_face=" << nb_face;
          for( Integer zz=0; zz<nb_face; ++zz ){
            ItemUniqueId fuid(connected_master_faces[face_index+zz]);
            //info() << " FACE=" << fuid;
            ni.addConnectedFace(fuid);
          }
        }
      }
      face_index += nb_face;
    }
  }
  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \warning Cette méthode ne doit pas être appelée en séquentiel.
 *
 * Envoie à tous les sous-domaine les faces de numéros uniques
 * \a faces_to_send de la liste \a face_map et réceptionne
 * celles de tous les autres sous-domaines.
 */
void TiedInterfaceBuilder::
_gatherFaces(ConstArrayView<ItemUniqueId> faces_to_send,
             TiedInterfaceFaceMap& face_map)
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  SerializeBuffer sbuf;
  sbuf.setMode(ISerializer::ModeReserve);
  Integer nb_to_send = faces_to_send.size();

  Int64UniqueArray unique_ids(nb_to_send);
  Int64UniqueArray cells_unique_ids(nb_to_send);
  Int64UniqueArray nodes_unique_id;
  nodes_unique_id.reserve(nb_to_send*4);
  IntegerUniqueArray nb_nodes(nb_to_send);
  RealUniqueArray coords;
  coords.reserve(3*nb_to_send);
  RealUniqueArray nodes_coords;
  nodes_coords.reserve(3*nb_to_send);
  for( Integer i=0; i<nb_to_send; ++i ){
    const TiedInterfaceFace& mf = face_map[faces_to_send[i]];
    for( Integer z=0, zs=mf.nbNode(); z<zs; ++z ){
      ItemUniqueId nuid(mf.nodeUniqueId(z));
      nodes_unique_id.add(nuid.asInt64());
      TiedInterfaceNodeInfo ni = m_nodes_info[nuid];
      nodes_coords.add(ni.m_coord.x);
      nodes_coords.add(ni.m_coord.y);
      nodes_coords.add(ni.m_coord.z);
    }
    unique_ids[i] = mf.uniqueId().asInt64();
    cells_unique_ids[i] = mf.cellUniqueId().asInt64();
    nb_nodes[i] = mf.nbNode();
    
    coords.add(mf.center().x);
    coords.add(mf.center().y);
    coords.add(mf.center().z);
  }
  sbuf.reserveInteger(1); // pour le nombre de faces
  sbuf.reserveInteger(1); // pour le numéro du sous-domaine
  sbuf.reserveInteger(1); // pour le nombre de noeuds dans la liste
  sbuf.reserve(DT_Int64,unique_ids.size()); // pour le unique id des faces 
  sbuf.reserve(DT_Int64,cells_unique_ids.size()); // pour le unique id des mailles des faces
  sbuf.reserveInteger(nb_nodes.size()); // pour le nombre de noeuds
  sbuf.reserve(DT_Int64,nodes_unique_id.size()); // pour la liste des noeuds
  sbuf.reserve(DT_Real,coords.size()); // pour les coordonnées du centre
  sbuf.reserve(DT_Real,nodes_coords.size()); // pour les coordonnées des noeuds
  sbuf.allocateBuffer();
  sbuf.setMode(ISerializer::ModePut);
  sbuf.putInteger(nb_to_send);
  sbuf.putInteger(my_rank);
  sbuf.putInteger(nodes_unique_id.size());
  sbuf.put(unique_ids);
  sbuf.put(cells_unique_ids); 
  sbuf.put(nb_nodes);
  sbuf.put(nodes_unique_id);
  sbuf.put(coords);
  sbuf.put(nodes_coords);
  
  SerializeBuffer recv_buf;
  pm->allGather(&sbuf,&recv_buf);
  recv_buf.setMode(ISerializer::ModeGet);

  for( Integer i=0; i<nb_rank; ++i ){
    Integer nb_face = recv_buf.getInteger();
    Integer sid = recv_buf.getInteger();
    Integer nb_node_unique_id = recv_buf.getInteger();
    //info() << " READ n=" << nb_face << " sid=" << sid;
    unique_ids.resize(nb_face);
    cells_unique_ids.resize(nb_face);
    nb_nodes.resize(nb_face);
    nodes_unique_id.resize(nb_node_unique_id);
    coords.resize(nb_face*3);
    nodes_coords.resize(nb_node_unique_id*3);

    recv_buf.get(unique_ids);
    recv_buf.get(cells_unique_ids);
    recv_buf.get(nb_nodes);
    recv_buf.get(nodes_unique_id);
    recv_buf.get(coords);
    recv_buf.get(nodes_coords);

    // Parcours toutes les faces reçues si certaines sont absentes,
    // les ajoute.
    Integer node_index = 0;
    for( Integer z=0; z<nb_face; ++z ){
      Integer nb_node = nb_nodes[z];
      ItemUniqueId uid(unique_ids[z]);
      ItemUniqueId cell_uid(cells_unique_ids[z]);
      if (!face_map.hasKey(uid)){
        Real3 center;
        center.x = coords[z*3];
        center.y = coords[z*3 + 1];
        center.z = coords[z*3 + 2];
        //info() << " RECV FACE: " << uid << " owner=" << sid << " center=" << center;

        Integer data_index = m_face_info_mng.size();
        for( Integer zz=0; zz<nb_node; ++zz ){
          ItemUniqueId nuid(nodes_unique_id[node_index+zz]);
          m_face_info_mng.add(nuid);
          // Ajoute le noeud à la liste.
          TiedInterfaceNodeInfo ni(nuid);
          ni.m_coord.x = nodes_coords[(node_index+zz)*3];
          ni.m_coord.y = nodes_coords[(node_index+zz)*3 + 1];
          ni.m_coord.z = nodes_coords[(node_index+zz)*3 + 2];
          //info() << "ADD NODE uid=" << nuid << " coord=" << ni.m_coord;
          m_nodes_info.add(nuid,ni);
        }
        TiedInterfaceFace sf(uid,cell_uid,nb_node,sid,data_index,&m_face_info_mng);
        sf.setCenter(center);
        face_map.add(uid,sf);
      }
      node_index += nb_node;
    }
  }
  pm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Migre les mailles sur les liaisons.
 *
 * Change le propriétaire de chaque maille liée à une face esclave
 * pour qu'il soit le même que celui de la maille maître associée.
 *
 * NOTE: que faire si une maille a plusieurs faces esclaves
 * connectées à des maîtres qui ne sont pas dans le même sous-domaine ?
 *
 * NOTE: version obsolete, car ne fonctionnant pas si une
 * maillage a plusieurs soudures
 */
void TiedInterfaceBuilder::
changeOwnersOld()
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  IntegerUniqueArray cells_nb_connected(cell_family->maxLocalId());
  cells_nb_connected.fill(0);
  VariableItemInt32 cells_owner = cell_family->itemsNewOwner();
  {
    ENUMERATE_FACE(iface,m_slave_interface.own()){
      Face face = *iface;
      const TiedInterfaceFace& slave_face = m_slave_faces[face.uniqueId()];
      ItemUniqueId master_face = m_slave_faces_master_face_uid[slave_face.uniqueId()];
      Integer master_owner = m_master_faces[master_face].owner();
      Cell slave_cell = face.cell(0);
      Int32 cell_lid = slave_cell.localId();
      ++cells_nb_connected[cell_lid];
      if (cells_nb_connected[cell_lid]>1){
        ARCANE_FATAL("Cell {0} is connected to more than one master face",
                     ItemPrinter(slave_cell));
      }
      cells_owner[slave_cell] = master_owner;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les liaisons entre mailles
 *
 * Change le propriétaire de chaque maille liée à une face esclave
 * pour qu'il soit le même que celui de la maille maître associée.
 *
 * Si une maille a plusieurs faces esclaves connectées à des maîtres,
 * s'assure que toutes ces maitres sont dans le meme sous-domaine.
 * Ajoute à \a linked_cells la liste des mailles liées et à
 * \a linked_owners le propriétaire associé.
 */
void TiedInterfaceBuilder::
changeOwners(Int64Array& linked_cells,Int32Array& linked_owners)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  UniqueArray<ItemUniqueId> cells_last_master_uid(cell_family->maxLocalId());
  cells_last_master_uid.fill(ItemUniqueId());
  //VariableItemInt32 cells_owner = cell_family->itemsNewOwner();
  {
    ENUMERATE_FACE(iface,m_slave_interface.own()){
      Face face = *iface;
      const TiedInterfaceFace& slave_face = m_slave_faces[face.uniqueId()];
      ItemUniqueId master_face = m_slave_faces_master_face_uid[slave_face.uniqueId()];
      Int32 master_owner = m_master_faces[master_face].owner();
      Cell slave_cell = face.cell(0);
      Int32 cell_lid = slave_cell.localId();
      ItemUniqueId last_master_uid = cells_last_master_uid[cell_lid];
      ItemUniqueId slave_face_cell_uid = face.cell(0).uniqueId();
      ItemUniqueId master_face_cell_uid = m_master_faces[master_face].cellUniqueId();
      // Le premier élément doit être le uid le plus petit
      if (slave_face_cell_uid<master_face_cell_uid){
        linked_cells.add(slave_face_cell_uid);
        linked_cells.add(master_face_cell_uid);
        linked_owners.add(face.owner());
      }
      else{
        linked_cells.add(master_face_cell_uid);
        linked_cells.add(slave_face_cell_uid);
        linked_owners.add(master_owner);
      }
      if (last_master_uid!=NULL_ITEM_UNIQUE_ID){
        // Maille connectée à plusieurs faces maîtres.
        // Vérifie qu'elles sont toutes dans le même sous-domaine.
        Int32 last_master_owner = m_master_faces[last_master_uid].owner();
        if (last_master_owner!=master_owner)
          ARCANE_FATAL("Cell {0} is connected to more than one master face:"
                       " face1={1} owner1={2} face2={3} owner2={4}", ItemPrinter(slave_cell),
                       last_master_uid , last_master_owner, master_face, master_owner);
      }
      cells_last_master_uid[cell_lid] = master_face;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit les infos sur une interface liée.
 *
 * Cette opération est collective si \a allow_communication est vrai.
 * Si \a allow_communication est faux, cela signifie que toutes
 * les faces esclaves d'une face maitre sont dans ce sous-domaine.
 */
void TiedInterfaceBuilder::
computeInterfaceConnections(bool allow_communication)
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();

  _computeMasterInterface();

  // En parallèle, il faut envoyer à tous le monde les infos sur la surface maître.
  if (allow_communication){
    UniqueArray<ItemUniqueId> master_faces_to_send;
    for( TiedInterfaceFaceMapEnumerator i(m_master_faces); ++i; ){
      TiedInterfaceFace& sf = *i;
      if (sf.owner()==my_rank)
        master_faces_to_send.add(sf.uniqueId());
    }
    _gatherFaces(master_faces_to_send,m_master_faces);
  }

#if 0
  if (allow_communication && arcaneIsCheck()){
    String fname = "master_face_";
    fname += m_slave_interface.name();
    if (is_parallel){
      fname += ".";
      fname += pm->commRank();
    }
    ofstream ofile(fname.localstr());
    _printFaces(ofile,m_master_faces);
  }
#endif

  {
    UniqueArray<ItemUniqueId> slave_faces_to_process;
    for( TiedInterfaceFaceMapEnumerator i(m_slave_faces); ++i; ){
      slave_faces_to_process.add((*i).uniqueId());
    }

    UniqueArray<ItemUniqueId> remaining_slave_faces;
    //TODO supprimer le 10000
    for( Integer zz=0; zz<10000; ++zz ){
      if (allow_communication){
        info() << " SEND RECV NODES INFO n=" << zz;
        _gatherAllNodesInfo();
      }
      //slave_faces_to_process.resize(remaining_slave_faces_index.size());
      //slave_faces_to_process.copy(remaining_slave_faces_index);
      //warning() << " SEARCH: NB=" << slave_faces_to_proceed.size();
      remaining_slave_faces.clear();
      _searchMasterFaces(slave_faces_to_process,remaining_slave_faces);
      Integer nb_remaining = remaining_slave_faces.size();
      Integer nb_to_process = slave_faces_to_process.size();
      if (allow_communication){
        info() << "NB REMAINING n=" << nb_remaining;
        nb_remaining = pm->reduce(Parallel::ReduceSum,nb_remaining);
        info() << "CUMULATIVE NB REMAINING n=" << nb_remaining;
        nb_to_process = pm->reduce(Parallel::ReduceSum,nb_to_process);
      }
      if (nb_remaining==0)
        break;
      if (nb_remaining==nb_to_process){
        ARCANE_FATAL("Can not compute master/slave infos for a tied interface. Remaining_slaves={0}",
                     nb_remaining);
      }
      slave_faces_to_process.copy(remaining_slave_faces);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceBuilder::
computeInterfaceInfos(TiedInterfaceBuilderInfos& infos,bool is_structured)
{
  _removeMasterFacesWithNoSlave();
  _computeProjectionInfos(infos,is_structured);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TiedInterface::m_is_debug = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterface::
TiedInterface(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_planar_tolerance(0.)
{
  if (!platform::getEnvironmentVariable("ARCANE_DEBUG_TIED_INTERFACE").null())
    m_is_debug = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterface::
~TiedInterface()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Groupe contenant les faces maîtres
FaceGroup TiedInterface::
masterInterface() const
{
  return m_master_interface;
}

//! Groupe contenant les faces esclaves
FaceGroup TiedInterface::
slaveInterface() const
{
  return m_slave_interface;
}

String TiedInterface::
masterInterfaceName() const
{
  return m_master_interface_name;
}

String TiedInterface::
slaveInterfaceName() const
{
  return m_slave_interface_name;
}

TiedInterfaceNodeList TiedInterface::
tiedNodes() const
{
  return m_tied_nodes;
}

TiedInterfaceFaceList TiedInterface::
tiedFaces() const
{
  return m_tied_faces;
}

void TiedInterface::
resizeNodes(IntegerConstArrayView new_sizes)
{
  m_tied_nodes.resize(new_sizes);
}

void TiedInterface::
resizeFaces(IntegerConstArrayView new_sizes)
{
  m_tied_faces.resize(new_sizes);
}

void TiedInterface::
setNodes(Integer index,ConstArrayView<TiedNode> nodes)
{
  for( Integer i=0, is=nodes.size(); i<is; ++i )
    m_tied_nodes[index][i] = nodes[i];
}

void TiedInterface::
setFaces(Integer index,ConstArrayView<TiedFace> faces)
{
  for( Integer i=0, is=faces.size(); i<is; ++i )
    m_tied_faces[index][i] = faces[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfacePartitionConstraint
: public TiedInterface::PartitionConstraintBase
{
 public:

  TiedInterfacePartitionConstraint(IMesh* mesh,ConstArrayView<FaceGroup> slave_interfaces,bool is_debug)
  : m_mesh(mesh), m_slave_interfaces(slave_interfaces), m_is_debug(is_debug), m_is_initial(true)
  {
  }

 public:

  virtual void addLinkedCells(Int64Array& linked_cells,Int32Array& linked_owners)
  {
    IParallelMng* pm = m_mesh->parallelMng();
    ITraceMng* tm = pm->traceMng();
    // Il faut faire les comms que pour le partitionnement initial car par la suite
    // les faces de part et d'autre d'une liaison sont dans le même sous-domaine.
    if (m_is_initial){
      Integer nb_interface = m_slave_interfaces.size();
      for( Integer i=0; i<nb_interface; ++i ){
        TiedInterfaceBuilder builder(m_mesh,m_slave_interfaces[i],false,m_is_debug);
        builder.computeInterfaceConnections(true);
        builder.changeOwners(linked_cells,linked_owners);
      }
    }
    else{
      TiedInterfaceCollection tied_interfaces = m_mesh->tiedInterfaces();
      for( TiedInterfaceCollection::Enumerator itied(tied_interfaces); ++itied; ){
        _addLinkedCells(*itied,linked_cells,linked_owners);
      }
    }
    tm->info() << "NB_LINKED_CELL=" << linked_cells.size();
  }

  virtual void setInitialRepartition(bool is_initial)
  {
    m_is_initial = is_initial;
  }

  void _addLinkedCells(ITiedInterface* interface,Int64Array& linked_cells,Int32Array& linked_owners)
  {
    FaceGroup master_interface = interface->masterInterface();
    //TiedInterfaceFaceList tied_faces = interface->tiedFaces();
    ENUMERATE_FACE(iface,master_interface){
      Face face = *iface;
      Cell cell = face.cell(0);
      Int64 cell_uid = cell.uniqueId();
      Int32 owner = cell.owner();
      for( Face isubface : face.slaveFaces() ){
        Cell sub_cell = isubface.cell(0);
        Int64 sub_cell_uid = sub_cell.uniqueId();
        if (sub_cell_uid<cell_uid){
          linked_cells.add(sub_cell_uid);
          linked_cells.add(cell_uid);
          linked_owners.add(owner);
        }
        else{
          linked_cells.add(cell_uid);
          linked_cells.add(sub_cell_uid);
          linked_owners.add(owner);
        }
      }
    }
  }

 private:

  IMesh* m_mesh;
  UniqueArray<FaceGroup> m_slave_interfaces;
  bool m_is_debug;
  bool m_is_initial;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé les informations pour l'interface soudée \a slave_interface/
 */
TiedInterface::PartitionConstraintBase* TiedInterface::
createConstraint(IMesh* mesh,ConstArrayView<FaceGroup> slave_interfaces)
{
  return new TiedInterfacePartitionConstraint(mesh,slave_interfaces,m_is_debug);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé les informations pour l'interface soudée \a slave_interface.
 */
void TiedInterface::
build(const FaceGroup& slave_interface,bool is_structured)
{
  info() << "Compute information for the tied interface " << slave_interface.name();

  TiedInterfaceBuilder builder(m_mesh,slave_interface,true,m_is_debug);
  TiedInterfaceBuilderInfos infos;

  builder.setPlanarTolerance(m_planar_tolerance);
  builder.computeInterfaceConnections(false);
  builder.computeInterfaceInfos(infos,is_structured);

  //infos.printInfos();

  m_slave_interface = slave_interface.own();
  m_slave_interface_name = slave_interface.name();
  m_master_interface = builder.masterInterface().own();
  m_master_interface_name = builder.masterInterface().name();

  {
    Integer nb_master_face = infos.m_master_faces_uid.size();
    m_tied_nodes.resize(infos.m_master_faces_nb_slave_node);
    m_tied_faces.resize(infos.m_master_faces_nb_slave_face);

    IItemFamily* node_family = m_mesh->itemFamily(IK_Node);
    IItemFamily* face_family = m_mesh->itemFamily(IK_Face);

    Int32UniqueArray slave_nodes_lid(infos.m_slave_nodes_uid.size());
    node_family->itemsUniqueIdToLocalId(slave_nodes_lid,infos.m_slave_nodes_uid);
    NodeInfoListView nodes(node_family);

    Int32UniqueArray slave_faces_lid(infos.m_slave_faces_uid.size());
    face_family->itemsUniqueIdToLocalId(slave_faces_lid,infos.m_slave_faces_uid);
    FaceInfoListView faces(face_family);

    for( Integer i_master=0; i_master<nb_master_face; ++i_master ){
      Integer nb_slave_node = infos.m_master_faces_nb_slave_node[i_master];
      Integer nb_slave_face = infos.m_master_faces_nb_slave_face[i_master];
      if (m_is_debug)
        info() << "Master Face: " << infos.m_master_faces_uid[i_master]
               << " nb_slave_face=" << nb_slave_face
               << " nb_slave_node=" << nb_slave_node
               << " node_index=" << infos.m_master_faces_slave_node_index[i_master];
      for( Integer zz=0; zz<nb_slave_node; ++zz ){
        Integer first_index = infos.m_master_faces_slave_node_index[i_master];
        
        if (m_is_debug)
          info() << "Node: uid=" <<  infos.m_slave_nodes_uid[first_index+zz]
                 << " uv=" << infos.m_slave_nodes_iso[first_index+zz];
        Integer local_id = slave_nodes_lid[first_index+zz];
        TiedNode tn(zz,nodes[local_id],infos.m_slave_nodes_iso[first_index+zz]);
        m_tied_nodes[i_master][zz] = tn;
      }

      for( Integer zz=0; zz<nb_slave_face; ++zz ){
        Integer first_index = infos.m_master_faces_slave_face_index[i_master];
        
        if (m_is_debug)
          info() << "Face: uid=" <<  infos.m_slave_faces_uid[first_index+zz];
        Integer local_id = slave_faces_lid[first_index+zz];
        m_tied_faces[i_master][zz] = TiedFace(zz,faces[local_id]);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterface::
setPlanarTolerance(Real tol)
{
  m_planar_tolerance = tol;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterface::
reload(IItemFamily* face_family,
       const String& master_interface_name,
       const String& slave_interface_name)
{
  m_master_interface_name = master_interface_name;
  m_slave_interface_name = slave_interface_name;
  ItemGroup group = face_family->findGroup(master_interface_name);
  if (group.null())
    ARCANE_FATAL("Can not find master group named '{0}'",master_interface_name);
  m_master_interface = group.own();
  group = face_family->findGroup(slave_interface_name);
  if (group.null())
    ARCANE_FATAL("Can not find slave group named '{0}'",slave_interface_name);
  m_slave_interface = group.own();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterface::
rebuild(ITiedInterfaceRebuilder* rebuilder,
        IntegerConstArrayView nb_slave_node,
        IntegerConstArrayView nb_slave_face)
{
  info() << "REBUILD_TIED_INTERFACE name=" << m_master_interface.name();
  m_tied_nodes.resize(nb_slave_node);
  m_tied_faces.resize(nb_slave_face);

  IItemFamily* node_family = m_mesh->itemFamily(IK_Node);
  IItemFamily* face_family = m_mesh->itemFamily(IK_Face);

  NodeInfoListView nodes(node_family);
  FaceInfoListView faces(face_family);

  Integer master_index = 0;
  Int32UniqueArray work_nodes_local_id;
  Real2UniqueArray work_nodes_iso;
  Int32UniqueArray work_faces_local_id;

  ENUMERATE_FACE(iface,m_master_interface){
    Face face = *iface;
    ArrayView<TiedNode> face_tied_nodes = m_tied_nodes[master_index];
    ArrayView<TiedFace> face_tied_faces = m_tied_faces[master_index];
    ++master_index;

    info(4) << "NEW VALUES face=" << ItemPrinter(face) << " n=" << face_tied_nodes.size() << " m=" << face_tied_faces.size();
    for( Node inode : face.nodes() )
      info(4) << "MasterFace node=" << ItemPrinter(inode);
    Integer nb_node = face_tied_nodes.size();
    Integer nb_face = face_tied_faces.size();
    work_nodes_local_id.resize(nb_node);
    work_nodes_iso.resize(nb_node);
    work_faces_local_id.resize(nb_face);

    rebuilder->fillTiedInfos(face,work_nodes_local_id,work_nodes_iso,work_faces_local_id);

    for( Integer zz=0; zz<nb_node; ++zz ){
      Integer local_id = work_nodes_local_id[zz];
      Real2 iso = work_nodes_iso[zz];
      info(4) << "NEW NODE slave_node=" << ItemPrinter(nodes[local_id]) << " iso=" << iso;
      face_tied_nodes[zz] = TiedNode(zz,nodes[local_id],iso);
    }
    
    for( Integer zz=0; zz<nb_face; ++zz ){
      Int32 local_id = work_faces_local_id[zz];
      info(4) << "NEW FACE slave_face=" << ItemPrinter(faces[local_id]);
      face_tied_faces[zz] = TiedFace(zz,faces[local_id]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterface::
checkValid()
{
  _checkValid(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterface::
_checkValid(bool is_print)
{
  Integer nb_error = 0;
  Integer max_print_error = 50;
  std::set<Int64> nodes_in_master_face;
  ITiedInterface* interface = this;
  FaceGroup slave_group = interface->slaveInterface();
  FaceGroup master_group = interface->masterInterface();
  info() << "Interface: Slave=" << slave_group.name()
         << " nb_face=" << slave_group.size();
  info() << "Interface: Master=" << master_group.name()
         << " nb_face=" << master_group.size();
  TiedInterfaceNodeList tied_nodes(interface->tiedNodes());
  TiedInterfaceFaceList tied_faces(interface->tiedFaces());
  ENUMERATE_FACE(iface,master_group){
    Face face = *iface;
    Int32 master_face_owner = face.owner();
    FaceVectorView slave_faces = face.slaveFaces();
    if (!face.isMasterFace()){
      ++nb_error;
      if (nb_error<max_print_error)
        error() << " Face uid=" << ItemPrinter(face) << " should have isMaster() true";
    }
    if (iface.index()>100000)
      break;
    nodes_in_master_face.clear();
    if (is_print)
      info() << "Master face uid=" << ItemPrinter(face)
             << " kind=" << face.kind()
             << " cell=" << ItemPrinter(face.cell(0))
             << " iface.index()=" << iface.index();
      
    Int32 cell_face_owner = face.cell(0).owner();
    if (cell_face_owner!=master_face_owner){
      ++nb_error;
      if (nb_error<max_print_error)
        error() << "master_face and its cell do not have the same owner: face_owner=" << master_face_owner
                << " cell_owner=" << cell_face_owner;
    }
    if (is_print)
      for( Node inode : face.nodes()  )
        info() << "Master face node uid=" << inode.uniqueId();
    for( Integer zz=0, zs=tied_nodes[iface.index()].size(); zz<zs; ++zz ){
      TiedNode tn = tied_nodes[iface.index()][zz];
      nodes_in_master_face.insert(tn.node().uniqueId());
      if (is_print){
        info() << " node_uid=" << tn.node().uniqueId()
               << " iso=" << tn.isoCoordinates()
               << " kind=" << tn.node().kind();
      }
    }
    for( Node inode : face.nodes() )
      if (nodes_in_master_face.find(inode.uniqueId())==nodes_in_master_face.end()){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << "node in master face not in slave node list node=" << ItemPrinter(inode);
      }
    Integer nb_tied = tied_faces[iface.index()].size();
    if (nb_tied!=slave_faces.size()){
      ++nb_error;
      if (nb_error<max_print_error){
        error() << "face=" << ItemPrinter(face) << " bad number of slave faces interne="
                << slave_faces.size() << " struct=" << nb_tied;
        ENUMERATE_FACE(islaveface,slave_faces){
          info() << "SLAVE " << ItemPrinter(*islaveface);
        }
      }
    }
    for( Integer zz=0, zs=tied_faces[iface.index()].size(); zz<zs; ++zz ){
      TiedFace tf = tied_faces[iface.index()][zz];
      Face tied_slave_face = tf.face();
      if (!tied_slave_face.isSlaveFace()){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << "slave face uid=" << ItemPrinter(tf.face()) << " should have isSlave() true";
      }
      if (tied_slave_face.masterFace()!=face){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << "slave face uid=" << ItemPrinter(tf.face()) << " should have masterSlave() valid";
      }
      if (tied_slave_face!=slave_faces[zz]){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << "bad slave face internal=" << ItemPrinter(slave_faces[zz])
                  << " struct=" << ItemPrinter(tied_slave_face);
      }
      Int32 slave_face_owner = tf.face().owner();
      if (slave_face_owner!=master_face_owner){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << "master_face and its slave_face do not have the same owner:"
                  << " master_face=" << ItemPrinter(face)
                  << " master_cell=" << ItemPrinter(face.cell(0))
                  << " slave_face=" << ItemPrinter(tf.face())
                  << " slave_cell=" << ItemPrinter(tf.face().cell(0));
      }
      if (is_print){
        info() << " face_uid=" << tf.face().uniqueId() << " cell=" << ItemPrinter(tf.face().cell(0));
      }
    }
  }

  if (nb_error!=0){
    ARCANE_FATAL("Errors in tied interface nb_error={0}",nb_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
