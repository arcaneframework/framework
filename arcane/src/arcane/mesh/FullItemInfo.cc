// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FullItemInfo.h                                              (C) 2000-2021 */
/*                                                                           */
/* Information de sérialisation d'une maille.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/FullItemInfo.h"

#include "arcane/ISerializer.h"
#include "arcane/ItemInternal.h"
#include "arcane/ItemInternalEnumerator.h"

#include "arcane/mesh/DynamicMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FullCellInfo::
FullCellInfo(Int64ConstArrayView cells_infos,Integer cell_index,
             ItemTypeMng* itm, Integer parent_info, 
             bool has_edge, bool has_amr, bool with_flags)
: m_nb_node(0)
, m_nb_edge(0)
, m_nb_face(0)
, m_first_edge(0)
, m_first_face(0)
, m_memory_used(0)
, m_type(0)
, m_first_parent_node(0)
, m_first_parent_edge(0)
, m_first_parent_face(0)
, m_first_parent_cell(0)
, m_parent_info(parent_info)
, m_has_edge(has_edge)
, m_has_amr(has_amr)
, m_with_flags(with_flags)
{
  Integer item_type_id = (Integer)cells_infos[cell_index];
  ItemTypeInfo* it = itm->typeFromId(item_type_id);
  m_type = it;
  _setInternalInfos();
  m_infos = Int64ConstArrayView(m_memory_used,&cells_infos[cell_index]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FullCellInfo::
print(ostream& o) const
{
  o << "Cell uid=" << uniqueId()
    << " nb_node=" << nbNode()
    << " nb_edge=" << nbEdge()
    << " nb_face=" << nbFace()
    << ' ';
  for( Integer z=0, zs=nbNode(); z<zs; ++z )
    o << " N" << z << "=" << nodeUniqueId(z);
  for( Integer z=0, zs=nbEdge(); z<zs; ++z )
    o << " E" << z << "=" << edgeUniqueId(z);
  for( Integer z=0, zs=nbFace(); z<zs; ++z )
    o << " F" << z << "=" << faceUniqueId(z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer FullCellInfo::
memoryUsed(ItemTypeInfo* it, Integer parent_info, bool has_edge, bool has_amr,bool with_flags)
{
  Integer memory_used = 0;
  // description de base d'une cellule
  memory_used += 3 + 2*it->nbLocalNode() + 2*it->nbLocalFace();
  if (has_edge)
    memory_used += 2*it->nbLocalEdge();
  // Facteur *2 : uid + type (tous sauf node)
  if (parent_info & PI_Node)
    memory_used += it->nbLocalNode();
  if (parent_info & PI_Edge)
    memory_used += 2*it->nbLocalEdge();
  if (parent_info & PI_Face)
    memory_used += 2*it->nbLocalFace();
  if (parent_info & PI_Cell)
    memory_used += 2;
  //! AMR
  if(has_amr)
    memory_used += 3;
  if(with_flags)
    memory_used += 1;

  return memory_used;
}

namespace
{
class SerializerDumpAdapter
{
 public:
  SerializerDumpAdapter(ISerializer* s) : m_serializer(s){}
  void put(Int64 v)
  {
    m_serializer->putInt64(v);
  }
 private:
  ISerializer* m_serializer;
};

class ArrayDumpAdapter
{
 public:
  ArrayDumpAdapter(Array<Int64>& a) : m_array_ref(a){}
  void put(Int64 v)
  {
    m_array_ref.add(v);
  }
 private:
  Array<Int64>& m_array_ref;
};

template<typename Adapter> void
_dumpCellInfo(ItemInternal* icell,Adapter buf, Integer parent_info,
              bool has_edge, bool has_amr,bool with_flags)
{
  Cell cell(icell);
  buf.put(cell.type());
  buf.put(cell.uniqueId().asInt64());
  buf.put(cell.owner());
  // Ajoute la liste des noeuds
  for( Item node : cell.nodes() ){
    buf.put(node.uniqueId().asInt64());
    buf.put(node.owner());
  }
  // Ajoute la liste des arêtes
  if (has_edge)
    for( Edge edge : cell.edges() ){
      buf.put(edge.uniqueId().asInt64());
      buf.put(edge.owner());
    }
  // Ajoute la liste des faces
  for( Face face : cell.faces() ){
    buf.put(face.uniqueId().asInt64());
    buf.put(face.owner());
  }
  if (parent_info & FullCellInfo::PI_Node) {
    for( Node node : cell.nodes() ){
      Item parent = node.parent(0);
      buf.put(parent.uniqueId().asInt64());
    }
  }
  if (parent_info & FullCellInfo::PI_Edge) {
    for( Edge edge : cell.edges() ){
      Item parent = edge.parent(0);
      buf.put(parent.uniqueId().asInt64());
      buf.put(parent.type());
    }
  }
  if (parent_info & FullCellInfo::PI_Face) {
    for( Face face : cell->faces() ){
      Item parent = face.parent(0);
      buf.put(parent.uniqueId().asInt64());
      buf.put(parent.type());
    }
  }
  if (parent_info & FullCellInfo::PI_Cell) {
    Item parent = cell.parent(0);
    buf.put(parent.uniqueId().asInt64());
    buf.put(parent.type());
  }
  //! AMR
  if(has_amr){
    buf.put(cell.level());
    if(cell.level() == 0){
      buf.put(NULL_ITEM_ID);
      buf.put(NULL_ITEM_ID);
    }
    else {
      Cell hParent = cell.hParent();
      buf.put(hParent.uniqueId().asInt64());
      buf.put(hParent.whichChildAmI(icell));
    }
  }
  if (with_flags)
    buf.put(icell->flags());
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FullCellInfo::
dump(ItemInternal* icell,ISerializer* buf, Integer parent_info,
     bool has_edge, bool has_amr,bool with_flags)
{
  SerializerDumpAdapter adapter(buf);
  _dumpCellInfo(icell,adapter,parent_info,has_edge,has_amr,with_flags);
}

void FullCellInfo::
dump(ItemInternal* icell,Array<Int64>& buf, Integer parent_info,
     bool has_edge, bool has_amr,bool with_flags)
{
  ArrayDumpAdapter adapter(buf);
  _dumpCellInfo(icell,adapter,parent_info,has_edge,has_amr,with_flags);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer FullCellInfo::
parentInfo(IMesh * mesh)
{
  Integer info = 0;
  if (mesh->cellFamily()->parentFamily())
    info |= PI_Cell;
  if (mesh->faceFamily()->parentFamily())
    info |= PI_Face;
  if (mesh->edgeFamily()->parentFamily())
    info |= PI_Edge;
  if (mesh->nodeFamily()->parentFamily())
    info |= PI_Node;
  return info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FullCellInfo::
_setInternalInfos()
{
  m_nb_node = m_type->nbLocalNode();
  if (m_has_edge)
    m_nb_edge = m_type->nbLocalEdge();
  else
    m_nb_edge = 0;
  m_nb_face = m_type->nbLocalFace();
  m_first_edge = 3 + m_nb_node*2;
  m_first_face = m_first_edge + m_nb_edge*2;
  m_first_parent_node = m_first_face + m_nb_face*2;
  m_first_parent_edge = m_first_parent_node;
  if (m_parent_info & PI_Node)
    m_first_parent_edge += m_nb_node;
  m_first_parent_face = m_first_parent_edge;
  if (m_parent_info & PI_Edge)
    m_first_parent_face += m_nb_edge*2;
  m_first_parent_cell = m_first_parent_face;
  if (m_parent_info & PI_Face)
    m_first_parent_cell += m_nb_face*2;
  m_memory_used = m_first_parent_cell;
  if (m_parent_info & PI_Cell)
    m_memory_used += 2;
  //! AMR
  if(m_has_amr){
    m_first_hParent_cell = m_memory_used;
    m_memory_used += 3;
  }
  if(m_with_flags)
    m_memory_used += 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
