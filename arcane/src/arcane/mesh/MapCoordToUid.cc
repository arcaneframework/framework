// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MapCoordToUid.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Recherche d'entités à partir de ses coordonnées.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/MapCoordToUid.h"

#include <limits>
#include <utility>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Real MapCoordToUid::TOLERANCE = 1.e-6;

//--------------------------------------------------------------------------
namespace
{
  // 10 bits per coordinate, to work with 32+ bit machines
  const unsigned int chunkmax = 1024;
  const unsigned long chunkmax2 = 1048576;
  const Real chunkfloat = 1024.0;
}

#ifdef ACTIVATE_PERF_COUNTER
const std::string MapCoordToUid::PerfCounter::m_names[] =
  {
    "Clear",
    "Fill",
    "Fill2",
    "Insert",
    "Find",
    "Key"
  };
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MapCoordToUid::Box::
Box()
{
  m_lower_bound = std::numeric_limits<Real>::max();
  m_upper_bound = -std::numeric_limits<Real>::max();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MapCoordToUid::Box::
init(IMesh* mesh)
{
  m_lower_bound = std::numeric_limits<Real>::max();
  m_upper_bound = -std::numeric_limits<Real>::max();
  // bounding box
  SharedVariableNodeReal3 nodes_coords(mesh->sharedNodesCoordinates());

  ENUMERATE_NODE(i_item,mesh->allNodes()){
    m_lower_bound[0] = std::min(m_lower_bound[0],nodes_coords[i_item].x);
    m_lower_bound[1] = std::min(m_lower_bound[1],nodes_coords[i_item].y);
    m_lower_bound[2] = std::min(m_lower_bound[2],nodes_coords[i_item].z);
    m_upper_bound[0] = std::max(m_upper_bound[0],nodes_coords[i_item].x);
    m_upper_bound[1] = std::max(m_upper_bound[1],nodes_coords[i_item].y);
    m_upper_bound[2] = std::max(m_upper_bound[2],nodes_coords[i_item].z);
  }
  // la box du maillage entier
  if (mesh->parallelMng()->isParallel()){
    mesh->parallelMng()->reduce(Parallel::ReduceMin,m_lower_bound);
    mesh->parallelMng()->reduce(Parallel::ReduceMax,m_upper_bound);
  }
  m_size = m_upper_bound - m_lower_bound;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MapCoordToUid::Box::
init2(IMesh* mesh)
{
  // bounding box
  m_lower_bound = std::numeric_limits<Real>::max();
  m_upper_bound = -std::numeric_limits<Real>::max();

  SharedVariableNodeReal3 nodes_coords(mesh->sharedNodesCoordinates());
  DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
  ItemInternalMap& nodes_map = dmesh->nodesMap();

  nodes_map.eachItem([&](Node node) {
    Int64 uid = node.uniqueId().asInt64();
    if(uid == NULL_ITEM_ID)
      return;
    m_lower_bound[0] = std::min(m_lower_bound[0],nodes_coords[node].x);
    m_lower_bound[1] = std::min(m_lower_bound[1],nodes_coords[node].y);
    m_lower_bound[2] = std::min(m_lower_bound[2],nodes_coords[node].z);
    m_upper_bound[0] = std::max(m_upper_bound[0],nodes_coords[node].x);
    m_upper_bound[1] = std::max(m_upper_bound[1],nodes_coords[node].y);
    m_upper_bound[2] = std::max(m_upper_bound[2],nodes_coords[node].z);
  });
  // la box du maillage entier
  if (mesh->parallelMng()->isParallel())  {
    mesh->parallelMng()->reduce(Parallel::ReduceMin,m_lower_bound);
    mesh->parallelMng()->reduce(Parallel::ReduceMax,m_upper_bound);
  }
  m_size = m_upper_bound - m_lower_bound ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MapCoordToUid::
MapCoordToUid(IMesh* mesh)
: m_mesh(mesh)
, m_box(NULL)
, m_nodes_coords(mesh->nodesCoordinates())
{
#ifdef ACTIVATE_PERF_COUNTER
  m_perf_counter.init() ;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
init()
{
  _clear();
	fill();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
init()
{
  _clear() ;
  fill();
  clearNewUids();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
init2()
{
  _clear();
  this->fill2();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NodeMapCoordToUid::
isItemToSuppress(Node node, const Int64 parent_uid) const
{
  for( Cell cell : node.cells() )
    if (cell.isActive() || cell.uniqueId()==parent_uid)
      return false;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
init2()
{
  _clear() ;
  this->fill2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 FaceMapCoordToUid::
faceCenter(Face face) const
{
  Real3 pfc = Real3::null();
  for( Node node : face.nodes() )
    pfc += m_nodes_coords[node];
  pfc /= static_cast<Real> (face.nbNode());
  return pfc ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceMapCoordToUid::
isItemToSuppress(Face face) const
{
  if (face.nbCell()==1)
    return ! face.cell(0).isActive() ;
  else{
    Cell cell0 = face.cell(0);
    Cell cell1 = face.cell(1);
    Integer level0 = cell0.level();
    Integer level1 = cell1.level();
    if(level0==level1)
      return ! (cell0.isActive() || cell1.isActive()) ;
    if(level0>level1)
      return ! cell0.isActive() ;
    else
      return ! cell1.isActive() ;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MapCoordToUid::
_clear()
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Clear) )

  for(map_type::iterator iter = m_map.begin();iter != m_map.end();++iter){
    iter->second.second = NULL_ITEM_ID ;
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::Clear) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
clearData(ArrayView<ItemInternal*> coarsen_cells)
{
  typedef std::set<Int64> set_type;
  typedef std::pair<set_type::iterator,bool> insert_return_type;
  set_type node_list;
  for (Integer icell = 0; icell < coarsen_cells.size(); icell++){
    Cell parent(coarsen_cells[icell]);
    for (UInt32 i = 0, nc = parent.nbHChildren(); i < nc; i++){
      Cell child = parent.hChild(i);
      for( Node node : child.nodes() ){
        Int64 uid = node.uniqueId() ;
        insert_return_type value = node_list.insert(uid) ;
        if(value.second){
          if(isItemToSuppress(node, parent.uniqueId())){
            m_mesh->traceMng()->debug(Trace::Highest)<<"SUPPRESS NODE : "<<uid<<" "<<m_nodes_coords[node] ;
            erase(m_nodes_coords[node]) ;
            //++count ;
          }
        }
      }
    }
  }
  //cout<<"NUMBER OF SUPPRESSED NODES : "<<count<<endl ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
clearData(ArrayView<ItemInternal*> coarsen_cells)
{
  typedef std::set<Int64> set_type;
  typedef std::pair<set_type::iterator,bool> insert_return_type;
  set_type face_list;
  for(Integer icell=0;icell<coarsen_cells.size();++icell){
    Cell cell(coarsen_cells[icell]);
    for (UInt32 i = 0, nc = cell.nbHChildren(); i < nc; i++){
      Cell child = cell.hChild(i) ;
      for( Face face : child.faces() ){
        Int64 uid = face.uniqueId() ;
        insert_return_type value = face_list.insert(uid) ;
        if(value.second){
          //cout<<" test face "<<uid<<" " ;
          //for(int ic=0;ic<iface->nbCell();++ic)
          //  cout<<" c["<<iface->cell(ic)->uniqueId()<<" "<<iface->cell(ic)->isActive()<<" "<<iface->cell(ic)->nbHChildren();
          //cout<<endl ;
          if(isItemToSuppress(face)){
            //Real3 fc = faceCenter(*iface) ;
            //cout<<"SUPPRESS FACE : "<<uid<<" "<<m_face_center[iface]<<endl ;
            erase(m_face_center[face]) ;
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
updateData(ArrayView<ItemInternal*> refine_cells)
{
  typedef std::set<Int64> set_type;
  typedef std::pair<set_type::iterator,bool> insert_return_type;
  set_type node_list;
  std::size_t count = 0;
  for (Integer icell = 0; icell < refine_cells.size(); icell++){
    Cell parent=refine_cells[icell];
    for (UInt32 i = 0, nc = parent.nbHChildren(); i < nc; i++){
      Cell child = parent.hChild(i) ;
      for( Node node : child.nodes() ){
        Int64 uid = node.uniqueId() ;
        insert_return_type value = node_list.insert(uid) ;
        if (value.second){
          bool is_new = insert(m_nodes_coords[node],uid) ;
          if(is_new){
            m_mesh->traceMng()->debug(Trace::Highest)<<"INSERT NODE : "<<uid<<" "<<m_nodes_coords[node] ;
            ++count;
          }
        }
      }
    }
  }
  m_mesh->traceMng()->debug(Trace::Highest)<<"NUMBER OF ADDED NODES : "<<count ;
  //check() ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
updateData(ArrayView<ItemInternal*> refine_cells)
{
  typedef std::set<Int64> set_type;
  typedef std::pair<set_type::iterator,bool> insert_return_type;
  set_type face_list ;
  for(Integer icell=0;icell<refine_cells.size();++icell){
    Cell cell = refine_cells[icell] ;
    for (UInt32 i = 0, nc = cell.nbHChildren(); i < nc; i++){
      Cell child = cell.hChild(i) ;
      for( Face face : child.faces() ){
        Int64 uid = face.uniqueId() ;
        insert_return_type value = face_list.insert(uid);
        if(value.second){
          Real3 fc = faceCenter(face);
          m_face_center[face] = fc;
          insert(fc,uid) ;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MapCoordToUid::
insert(const Real3 p,const Int64 uid,Real tol)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Insert) )
  //this->m_map.insert(std::make_pair(this->key(p), std::make_pair(p,uid)));
  Int64 pointkey = this->key(p);

  // Look for the exact key first
  std::pair<map_type::iterator,map_type::iterator>
  pos = m_map.equal_range(pointkey);
  map_type::iterator iter = pos.first ;
  while (iter != pos.second){
    if ( areClose(p,iter->second.first,tol)){
      Int64 old_uid = iter->second.second;
      iter->second.second = uid ;
      CHECKPERF( m_perf_counter.stop(PerfCounter::Insert) )
      return old_uid ;
    }
    else
      ++iter;
  }
  // Look for neighboring bins' keys next
  for (int xoffset = -1; xoffset != 2; ++xoffset)
    for (int yoffset = -1; yoffset != 2; ++yoffset)
      for (int zoffset = -1; zoffset != 2; ++zoffset) {
        std::pair<map_type::iterator,map_type::iterator>
        pos2 = m_map.equal_range(pointkey +
                                 xoffset*chunkmax2 +
                                 yoffset*chunkmax +
                                 zoffset);
        map_type::iterator iter2 = pos2.first ;
        while (iter2 != pos2.second){
          if ( areClose(p,iter2->second.first,tol)){
            Int64 old_uid = iter2->second.second ;
            iter2->second.second = uid ;
            CHECKPERF( m_perf_counter.stop(PerfCounter::Insert) )
            return old_uid;
          }
          else
            ++iter2;
        }
      }
  m_map.insert(pos.first,std::make_pair(pointkey, std::make_pair(p,uid)));
  CHECKPERF( m_perf_counter.stop(PerfCounter::Insert) )
  return NULL_ITEM_ID ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MapCoordToUid::
find(const Real3 p,const Real tol)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Find) )

	// Look for a likely key in the multimap
	Int64 pointkey = this->key(p);

	// Look for the exact key first
	std::pair<map_type::iterator,map_type::iterator>
	pos = m_map.equal_range(pointkey);

	while (pos.first != pos.second)
		if ( areClose(p,pos.first->second.first,tol)){
			//debug() << "find(),MapCoordToUid";
			return pos.first->second.second;
		}
		else
			++pos.first;

	// Look for neighboring bins' keys next
	for (int xoffset = -1; xoffset != 2; ++xoffset)
		for (int yoffset = -1; yoffset != 2; ++yoffset)
			for (int zoffset = -1; zoffset != 2; ++zoffset){
				std::pair<map_type::iterator,map_type::iterator>
				pos = m_map.equal_range(pointkey +
                                xoffset*chunkmax2 +
                                yoffset*chunkmax +
                                zoffset);
				while (pos.first != pos.second){
					if ( areClose(p,pos.first->second.first,tol)){
						return pos.first->second.second;
					}
					else
						++pos.first;
				}
			}

	CHECKPERF( m_perf_counter.stop(PerfCounter::Find) )
	return NULL_ITEM_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MapCoordToUid::
erase(const Real3 p,const Real tol)
{
  ARCANE_UNUSED(tol);
  // Look for a likely key in the multimap
  insert(p,NULL_ITEM_ID);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MapCoordToUid::
key(const Real3 p)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Key) )
	Real xscaled = (p.x - m_box->m_lower_bound.x) / (m_box->m_size.x),
	yscaled = (p.y - m_box->m_lower_bound.y) /	(m_box->m_size.y),
	zscaled = (m_box->m_upper_bound.z != m_box->m_lower_bound.z)
    ? ((p.z - m_box->m_lower_bound.z)/(m_box->m_size.z)) : p.z;
#ifndef NO_USER_WARNING
#warning [MapCoordToUid::key] 2D m_box->m_upper_bound.z==m_box->m_lower_bound.z
#endif
	Int64 n0 = static_cast<Int64> (chunkfloat * xscaled),
	n1 = static_cast<Int64> (chunkfloat * yscaled),
	n2 = static_cast<Int64> (chunkfloat * zscaled);

	CHECKPERF( m_perf_counter.stop(PerfCounter::Key) )
	return chunkmax2*n0 + chunkmax*n1 + n2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
fill()
{
  // Populate the nodes map
  CHECKPERF( m_perf_counter.start(PerfCounter::Fill) )
  m_mesh->traceMng()->debug(Trace::Highest)<<"[MapCoordToUid::fill] nb allNodes="<<m_mesh->allNodes().size();
  ENUMERATE_NODE(i_item,m_mesh->allNodes()){
    Node node = *i_item;
    Int64 uid = node.uniqueId().asInt64();
    this->insert(m_nodes_coords[i_item],uid);
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::Fill) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
check()
{
  m_mesh->traceMng()->debug(Trace::Highest)<<"[NODE MapCoordToUid::fill] nb allNodes="<<m_mesh->allNodes().size();
  // Populate the nodes map
  std::set<Int64> set;
  ENUMERATE_NODE(i_item,m_mesh->allNodes()){
    Node node = *i_item;
    Int64 uid = node.uniqueId().asInt64();
    m_mesh->traceMng()->debug(Trace::Highest)<<"\t[NODE MapCoordToUid::fill] node_"<<node.localId()<<", uid="<<uid<<" "<<m_nodes_coords[i_item];
    Int64 map_uid = find(m_nodes_coords[i_item]);
    set.insert(uid) ;
    if(uid!=map_uid){
      m_mesh->traceMng()->error()<<"MAP NODE ERROR : uid = "<<uid<<" coords="<<m_nodes_coords[i_item];
      m_mesh->traceMng()->fatal()<<"MAP NODE ERROR : "<<map_uid<<" found, expected uid "<<uid;
    }
  }
  {
    Integer count = 0 ;
    for(map_type::iterator iter = m_map.begin();iter!=m_map.end();++iter)
      if(iter->second.second!=NULL_ITEM_ID){
        ++count ;
        if(set.find(iter->second.second)==set.end()){
          m_mesh->traceMng()->fatal()<<"MAP NODE ERROR : node "<<iter->second.second<<" "<<iter->second.first<<" does not exist";
        }
      }
    if(count !=m_mesh->allNodes().size())
      m_mesh->traceMng()->fatal()<<"MAP NODE ERROR : map size"<<count<<" != "<<m_mesh->allNodes().size();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
fill()
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Fill) )
  m_mesh->traceMng()->debug(Trace::Highest)<<"[MapCoordToUid::fill] nb allFaces="<<m_mesh->allFaces().size();
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    this->insert(m_face_center[iface],iface->uniqueId());
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::Fill) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
check()
{
  m_mesh->traceMng()->debug(Trace::Highest)<<"[FACE MapCoordToUid::fill] nb allFaces="<<m_mesh->allFaces().size();
  std::set<Int64> set;
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    Int64 uid = iface->uniqueId() ;
    m_mesh->traceMng()->debug(Trace::Highest)<<"\t[FACE MapCoordToUid::fill] face_"<<iface->localId()<<", uid="<<uid;
    for( Node inode : iface->nodes() ){
      m_mesh->traceMng()->debug(Trace::Highest)<<"\t\t[FACE MapCoordToUid::fill] node_"<<inode.localId();
    }
    Int64 map_uid = find(m_face_center[iface]);
    set.insert(uid) ;
    if(uid!=map_uid){
      m_mesh->traceMng()->fatal()<<"MAP FACE ERROR : "<<map_uid<<" found, expected uid "<<uid;
    }
  }
  {
    Integer count = 0;
    for(map_type::iterator iter = m_map.begin();iter!=m_map.end();++iter)
      if(iter->second.second!=NULL_ITEM_ID){
        ++count;
        if(set.find(iter->second.second)==set.end()){
          m_mesh->traceMng()->fatal()<<"MAP FACE ERROR : node "<<iter->second.second<<" "<<iter->second.first<<" does not exist";
        }
      }
    if(count !=m_mesh->allFaces().size())
      m_mesh->traceMng()->fatal()<<"MAP FACE ERROR : map size"<<count<<" != "<<m_mesh->allNodes().size();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
fill2()
{
  CHECKPERF( m_perf_counter.start(PerfCounter::Fill2) )
  // Populate the nodes map
  DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(m_mesh));
  ItemInternalMap& nodes_map = dmesh->nodesMap();
  nodes_map.eachItem([&](Node node) {
    Int64 uid = node.uniqueId().asInt64();
    if(uid == NULL_ITEM_ID)
      return;
    this->insert(m_nodes_coords[node],uid);
  });
  CHECKPERF( m_perf_counter.stop(PerfCounter::Fill2) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NodeMapCoordToUid::
check2()
{
  // Populate the nodes map
  DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(m_mesh));
  ItemInternalMap& nodes_map = dmesh->nodesMap();
  nodes_map.eachItem([&](Node node) {
    Int64 uid = node.uniqueId().asInt64();
    Int64 map_uid = find(m_nodes_coords[node]);
    if(uid!=map_uid)
      ARCANE_FATAL("MAP NODE ERROR : '{0}' found, expected uid '{1}'",map_uid,uid);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
fill2()
{
  CHECKPERF( m_perf_counter.stop(PerfCounter::Fill2) )
  DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(m_mesh));
  ItemInternalMap& faces_map = dmesh->facesMap();
  faces_map.eachItem([&](Face face) {
    Int64 face_uid = face.uniqueId().asInt64();
    if(face_uid == NULL_ITEM_ID)
      return;
    this->insert(faceCenter(face),face_uid);
  });
  CHECKPERF( m_perf_counter.stop(PerfCounter::Fill2) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
check2()
{
  DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(m_mesh));
  ItemInternalMap& faces_map = dmesh->facesMap();
  faces_map.eachItem([&](Face face) {
    Int64 face_uid = face.uniqueId().asInt64();
    Int64 map_uid = find(faceCenter(face));
    if (face_uid != map_uid)
      ARCANE_FATAL("MAP NODE ERROR : '{0}' found, expected uid '{1}'",map_uid,face_uid);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
initFaceCenter()
{
  ENUMERATE_FACE(iface,m_mesh->allFaces()){
    m_face_center[iface] = faceCenter(*iface);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceMapCoordToUid::
updateFaceCenter(ArrayView<ItemInternal*> refine_cells)
{
  typedef std::set<Int64> set_type ;
  typedef std::pair<set_type::iterator,bool> insert_return_type;
  set_type face_list ;
  for(Integer icell=0;icell<refine_cells.size();++icell){
    Cell cell = refine_cells[icell] ;
    for (UInt32 i = 0, nc = cell.nbHChildren(); i<nc; ++i ){
      Cell child = cell.hChild(i) ;
      for( Face face : child.faces() ){
        Int64 uid = face.uniqueId() ;
        insert_return_type value = face_list.insert(uid);
        if (value.second){
          Real3 fc = faceCenter(face);
          m_face_center[face] = fc;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
