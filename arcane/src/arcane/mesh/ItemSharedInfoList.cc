// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfoList.cc                                       (C) 2000-2020 */
/*                                                                           */
/* Liste de 'ItemSharedInfo'.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/VariableTypes.h"

#include "arcane/mesh/ItemSharedInfoList.h"
#include "arcane/mesh/ItemFamily.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemSharedInfoList::ItemNumElements
{
 public:
  ItemNumElements(Integer type,Integer nb_edge,Integer nb_face,Integer nb_cell,
                  Integer edge_allocated,Integer face_allocated,Integer cell_allocated)
  : m_type(type), m_nb_edge(nb_edge), m_nb_face(nb_face), m_nb_cell(nb_cell),
    m_edge_allocated(edge_allocated), m_face_allocated(face_allocated), m_cell_allocated(cell_allocated)
    //! AMR
    , m_nb_hParent(0),m_nb_hChildren(0), m_hParent_allocated(0),m_hChild_allocated(0)
    {}
  //! AMR
  ItemNumElements(Integer type,Integer nb_edge,Integer nb_face,Integer nb_cell,
		          Integer nb_parent, Integer nb_children,
                  Integer edge_allocated,Integer face_allocated,Integer cell_allocated,
                  Integer parent_allocated, Integer child_allocated)
    : m_type(type), m_nb_edge(nb_edge), m_nb_face(nb_face), m_nb_cell(nb_cell),
      m_edge_allocated(edge_allocated), m_face_allocated(face_allocated), m_cell_allocated(cell_allocated),
      m_nb_hParent(nb_parent),m_nb_hChildren(nb_children),
      m_hParent_allocated(parent_allocated),m_hChild_allocated(child_allocated) {}

 public:

  static bool m_debug;

  Integer m_type;
  Integer m_nb_edge;
  Integer m_nb_face;
  Integer m_nb_cell;
  Integer m_edge_allocated;
  Integer m_face_allocated;
  Integer m_cell_allocated;
  //! AMR
  Integer m_nb_hParent;
  Integer m_nb_hChildren;
  Integer m_hParent_allocated;
  Integer m_hChild_allocated;

  bool operator<(const ItemNumElements& b) const
    {
#ifdef ARCANE_CHECK
      if (m_debug){
        cout << "Compare:\nTHIS=";
        print(cout);
        cout << "\nTO=";
        b.print(cout);
        cout << "\nVAL=" << compare(b);
        cout << "\n";
      }
#endif
      return compare(b);
    }
 private:
  inline bool compare(const ItemNumElements& b) const
    {
      if (m_type!=b.m_type)
        return (m_type<b.m_type);

      if (m_nb_edge!=b.m_nb_edge)
        return (m_nb_edge<b.m_nb_edge);

      if (m_nb_face!=b.m_nb_face)
        return (m_nb_face<b.m_nb_face);

      if (m_nb_cell!=b.m_nb_cell)
        return (m_nb_cell<b.m_nb_cell);

      if (m_edge_allocated!=b.m_edge_allocated)
        return (m_edge_allocated<b.m_edge_allocated);

      if (m_face_allocated!=b.m_face_allocated)
        return (m_face_allocated<b.m_face_allocated);
//! AMR
      if (m_nb_hParent!=b.m_nb_hParent)
          return (m_nb_hParent<b.m_nb_hParent);
      if (m_nb_hChildren!=b.m_nb_hChildren)
          return (m_nb_hChildren<b.m_nb_hChildren);
      if (m_hParent_allocated!=b.m_hParent_allocated)
          return (m_hParent_allocated<b.m_hParent_allocated);
      if (m_hChild_allocated!=b.m_hChild_allocated)
          return (m_hChild_allocated<b.m_hChild_allocated);

      return m_cell_allocated<b.m_cell_allocated;
    }
 public:
  void print(ostream& o) const
    {
      o << " Type=" << m_type
        << " Edge=" << m_nb_edge
        << " Face=" << m_nb_face
        << " Cell=" << m_nb_cell
        << " EdgeAlloc=" << m_edge_allocated
        << " FaceAlloc=" << m_face_allocated
        << " CellAlloc=" << m_cell_allocated
//! AMR
      << " Parent=" << m_nb_hParent
      << " Child=" << m_nb_hChildren
      << " ParentAlloc=" << m_hParent_allocated
      << " ChildAlloc=" << m_hChild_allocated
      ;
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemSharedInfoList::ItemNumElements::m_debug = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ostream& operator<<(ostream& o,const ItemSharedInfoList::ItemNumElements& v)
{
  v.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemSharedInfoList::Variables
{
 public:
  Variables(IMesh* mesh,const String& name)
  : m_infos_values(VariableBuildInfo(mesh,name)){}
 public:
  VariableArray2Int32 m_infos_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoList::
ItemSharedInfoList(ItemFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
, m_nb_item_shared_info(0)
, m_item_kind(family->itemKind())
, m_item_shared_infos_buffer(new MultiBufferT<ItemSharedInfo>(100))
, m_infos_map(new ItemSharedInfoMap())
, m_variables(0)
, m_list_changed(false)
, m_connectivity_info_changed(true)
, m_max_node_per_item(0)
, m_max_edge_per_item(0)
, m_max_face_per_item(0)
, m_max_cell_per_item(0)
, m_max_node_per_item_type(0)
, m_max_edge_per_item_type(0)
, m_max_face_per_item_type(0)
{
  {
    String var_name(family->name());
    var_name = var_name + "_SharedInfoList";
    m_variables = new Variables(family->mesh(),var_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoList::
~ItemSharedInfoList()
{
  delete m_variables;
  delete m_infos_map;
  delete m_item_shared_infos_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
prepareForDump()
{
  Integer n = m_item_shared_infos.size();
  info(4) << "ItemSharedInfoList: write: " << m_family->name()
          << " count=" << n << " changed=" << m_list_changed;
  log() << "ItemSharedInfoList: write: " << m_family->name()
        << " count=" << n << " changed=" << m_list_changed;

  if (!m_list_changed)
    return;
  m_list_changed = false;
  //Integer n = m_item_shared_infos.size();
  Integer element_size = ItemSharedInfo::serializeSize();
  m_variables->m_infos_values.resize(n,element_size);
  for( Integer i=0; i<n; ++i ){
    m_item_shared_infos[i]->serializeWrite(m_variables->m_infos_values[i]);
    //     if (i<20 && m_family->itemKind()==IK_Particle){
    //       ItemSharedInfo* isi = m_item_shared_infos[i];
    //       log() << "FAMILY" << m_family->name() << " ISI: i=" << i << " values=" << *isi;
    //       info() << "FAMILY" << m_family->name() << "ISI: i=" << i << " values=" << *isi;
    //     }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
readFromDump()
{
  Integer element_size = ItemSharedInfo::serializeSize();
  Integer n = m_variables->m_infos_values.dim1Size();
  info() << "ItemSharedInfoList: read: " << m_family->name() << " count=" << n;

  if (n>0){
    // Le nombre d'éléments sauvés dépend de la version de Arcane et du fait
    // qu'on utilise ou pas l'AMR.
    Integer stored_size = m_variables->m_infos_values[0].size();
    if (stored_size==ItemSharedInfo::serializeAMRSize()){
    }
    else if (stored_size!=element_size)
      ARCANE_FATAL("Incoherence of saved data (most probably due to a"
                   " difference of versions between the protection and the executable."
                   " stored_size={0} element_size={1} count={2}",
                   stored_size,element_size,n);
  }

  // Tous les types vont a nouveau être ajoutés à la liste
  m_item_shared_infos.clear();
  m_infos_map->clear();

  if (n==0)
    return;

  for( Integer i=0; i<n; ++i )
    allocOne();

  MeshItemInternalList* miil = m_family->mesh()->meshItemInternalList();
  ItemInternalConnectivityList* iicl = m_family->itemInternalConnectivityList();

  ItemTypeMng* itm = m_family->mesh()->itemTypeMng();
  for( Integer i=0; i<n; ++i ){
    Int32ConstArrayView buffer(m_variables->m_infos_values[i]);
    // Le premier élément du tampon contient toujours le type de l'entité
    ItemTypeInfo* it = itm->typeFromId(buffer[0]);
    ItemSharedInfo* isi = m_item_shared_infos[i];
    *isi = ItemSharedInfo(m_family,it,miil,iicl,m_family->uniqueIds(),buffer);

    ItemNumElements ine(it->typeId(),isi->nbEdge(),isi->nbFace(),isi->nbCell(),
                        isi->nbHParent(),isi->nbHChildren(),
                        isi->edgeAllocated(),isi->faceAllocated(),isi->cellAllocated(),
                        isi->hParentAllocated(),isi->hChildAllocated()
                        );
    std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));
    if (!old.second){
      // Vérifie que l'instance ajoutée ne remplace pas une instance déjà présente,
      // auquel il s'agit d'une erreur interne (opérateur de comparaison foireux)
      dumpSharedInfos();
      ItemNumElements::m_debug = true;
      bool compare = m_infos_map->find(ine)!=m_infos_map->end();
      fatal() << "INTERNAL: ItemSharedInfoList::readfromDump(): SharedInfo already present family=" << m_family->name()
              << "\nWanted:"
              << " type=" << it->typeId()
              << " edge=" << isi->nbEdge()
              << " face=" << isi->nbFace()
              << " cell=" << isi->nbCell()
              << " nb_hparent=" << isi->nbHParent()
              << " nb_hchildren=" << isi->nbHChildren()
              << " edge_alloc=" << isi->edgeAllocated()
              << " face_alloc=" << isi->faceAllocated()
              << " cell_alloc=" << isi->cellAllocated()
              << " haparent_alloc=" << isi->hParentAllocated()
              << " hchildren_alloc=" << isi->hChildAllocated()
              << " compare=" << compare
              << "\nNEW_INE=(" << ine << ")"
              << "\nOLD_INE=(" << old.first->first << ")"
              << "\nNEW_ISI=(" << *isi << ")"
              << "\nOLD_ISI=(" << *old.first->second << ")";
     }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
checkValid()
{
  bool has_error = false;

  info() << "ItemSharedInfoList: check valid family=" << m_family->name()
         << " count=" << m_nb_item_shared_info
         << " list=" << m_item_shared_infos.size()
         << " free=" << m_free_item_shared_infos.size()
         << " changed=" << m_list_changed;

  // Premièrement, le item->localId() doit correspondre à l'indice
  // dans le tableau m_internal
  for( Integer i=0, is=m_item_shared_infos.size(); i<is; ++i ){
    ItemSharedInfo* item = m_item_shared_infos[i];
    if (item->index()!=i){
      error() << "The index (" << item->index() << ") from the list 'ItemSharedInfo' "
              << "of the family " << m_family->name() << " is not "
              << "coherent with its internal value (" << i << ")";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Internal error with the mesh structure");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfo* ItemSharedInfoList::
findSharedInfo(ItemTypeInfo* type)
{
  Integer nb_edge = 0;
  Integer nb_face = 0;
  Integer nb_cell = 0;
  Integer nb_parent = 0;
  Integer nb_children = 0;
  Integer edge_allocated = 0;
  Integer face_allocated = 0;
  Integer cell_allocated = 0;
  Integer parent_allocated = 0;
  Integer child_allocated = 0;

  ItemNumElements ine(type->typeId(),nb_edge,nb_face,nb_cell,
                      nb_parent,nb_children,
                      edge_allocated,face_allocated,cell_allocated,
                      parent_allocated,child_allocated);
  ItemSharedInfoMap::const_iterator i = m_infos_map->find(ine);
  if (i!=m_infos_map->end())
    return i->second;
  MeshItemInternalList* miil = m_family->mesh()->meshItemInternalList();
  ItemInternalConnectivityList* iicl = m_family->itemInternalConnectivityList();
  // Infos pas trouvé. On en construit une nouvelle
  ItemSharedInfo* isi = allocOne();
  Integer old_index = isi->index();
  *isi = ItemSharedInfo(m_family,type,miil,iicl,m_family->uniqueIds(),nb_edge,nb_face,nb_cell,
                        nb_parent,nb_children,
                        edge_allocated,face_allocated,cell_allocated,
                        parent_allocated,child_allocated);
  isi->setIndex(old_index);
  //isi->m_infos = m_items_infos.begin();
  std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));

  //#ifdef ARCANE_CHECK
  if (!old.second){
    // Vérifie que l'instance ajoutée ne remplace pas une instance déjà présente,
    // auquel il s'agit d'une erreur interne (opérateur de comparaison foireux)
    dumpSharedInfos();
    ItemNumElements::m_debug = true;
    bool compare = m_infos_map->find(ine)!=m_infos_map->end();
    fatal() << "INTERNE: ItemSharedInfoList::findSharedInfo() SharedInfo déjà présent\n"
            << "\nWanted:"
            << " type=" << type->typeId()
            << " edge=" << nb_edge
            << " face=" << nb_face
            << " cell=" << nb_cell
            << " edge_alloc=" << edge_allocated
            << " face_alloc=" << face_allocated
            << " cell_alloc=" << cell_allocated
            << " compare=" << compare
            << "\nNEW_INE=(" << ine << ")"
            << "\nOLD_INE=(" << old.first->first << ")"
            << "\nNEW_ISI=(" << *isi << ")"
            << "\nOLD_ISI=(" << *old.first->second << ")";
  }
  //#endif
  return isi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
dumpSharedInfos()
{
  info() << "--- ItemSharedInfos: family=" << m_family->name();
  for( ConstIterT<ItemSharedInfoMap> i(*m_infos_map); i(); ++i ){
    info() << "INE: " << i->first;
    info() << "ISI: " << *i->second;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
setSharedInfosPtr(Int32* ptr)
{
  for( ConstIterT<ItemSharedInfoMap> i(*m_infos_map); i(); ++i ){
    ItemSharedInfo* isi = i->second;
    isi->_setInfos(ptr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxNodePerItem()
{
  _checkConnectivityInfo();
  return m_max_node_per_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxEdgePerItem()
{
  _checkConnectivityInfo();
  return m_max_edge_per_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxFacePerItem()
{
  _checkConnectivityInfo();
  return m_max_face_per_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxCellPerItem()
{
  _checkConnectivityInfo();
  return m_max_cell_per_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxLocalNodePerItemType()
{
  _checkConnectivityInfo();
  return m_max_node_per_item_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxLocalEdgePerItemType()
{
  _checkConnectivityInfo();
  return m_max_edge_per_item_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemSharedInfoList::
maxLocalFacePerItemType()
{
  _checkConnectivityInfo();
  return m_max_face_per_item_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
_checkConnectivityInfo()
{
  if (!m_connectivity_info_changed)
    return;

  m_max_node_per_item = 0;
  m_max_edge_per_item = 0;
  m_max_face_per_item = 0;
  m_max_cell_per_item = 0;
  m_max_node_per_item_type = 0;
  m_max_edge_per_item_type = 0;
  m_max_face_per_item_type = 0;

  for( ConstIterT<ItemSharedInfoMap> i(*m_infos_map); i(); ++i ){
    ItemSharedInfo* isi = i->second;
    m_max_node_per_item = math::max(m_max_node_per_item,isi->nbNode());
    m_max_edge_per_item = math::max(m_max_edge_per_item,isi->nbEdge());
    m_max_face_per_item = math::max(m_max_face_per_item,isi->nbFace());
    m_max_cell_per_item = math::max(m_max_cell_per_item,isi->nbCell());
    m_max_node_per_item_type = math::max(m_max_node_per_item_type,
                                         isi->m_item_type->nbLocalNode());
    m_max_edge_per_item_type = math::max(m_max_edge_per_item_type,
                                         isi->m_item_type->nbLocalEdge());
    m_max_face_per_item_type = math::max(m_max_face_per_item_type,
                                         isi->m_item_type->nbLocalFace());
  }

  m_connectivity_info_changed = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
