// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchange.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Echange un maillage entre entre sous-domaines.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iterator.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Collection.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/Item.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IItemFamilyNetwork.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"

#include "arcane/mesh/IndexedItemConnectivityAccessor.h"
#include "arcane/mesh/MeshExchange.h"
#include "arcane/mesh/NewItemOwnerBuilder.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class MeshExchange::IncrementalUnorderedMultiArray 
{
 public:
 
  IncrementalUnorderedMultiArray(Integer size) 
  : m_current_increment(-1), m_current_size(0), m_index(size, -1),
    m_size(size, 0)
  {
  }

  IncrementalUnorderedMultiArray()
  : m_current_increment(-1), m_current_size(0) {}
  
  inline void resize(Integer size)
  {
    m_index.resize(size, -1);
    m_size.resize(size, 0);
  }
  
  inline Integer size() const
  {
    return m_index.size();
  }
  
  inline void dataReserve(Integer size)
  {
    m_data.reserve(size);
  }
  
  inline void addData(const T data)
  {
    m_data.add(data);
    m_current_size++;
  }
  
  inline Int32 index(Integer id) const
  {
    return m_index[id];
  }
  
  inline void beginIncrement(Integer id)
  {
    ARCANE_ASSERT((m_current_increment == -1),("call endIncrement before begin"));
    m_current_increment = id;
    m_current_size = 0;
    m_index[m_current_increment] = m_data.size();
  }
  
  inline void endIncrement()
  {
    m_size[m_current_increment] = m_current_size;
    m_current_increment = -1;
    m_current_size = 0;
  }
    
  inline ArrayView<T> at(Integer id)
  {
    return m_data.subView(m_index[id],m_size[id]);
  }
  
  inline ConstArrayView<T> at(Integer id) const
  {
    return m_data.subView(m_index[id],m_size[id]);
  }

 private:
  Integer m_current_increment;
  Integer m_current_size;
  Int32UniqueArray m_index;
  Int32UniqueArray m_size;
  UniqueArray<T> m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class MeshExchange::DynamicMultiArray 
{
public:
 
  DynamicMultiArray(Integer size) 
    : m_data(size) {}
 
  DynamicMultiArray() {}
  
  inline void resize(Integer size) {
    m_data.resize(size);
  }
  
  inline Integer size() const {
    return m_data.size();
  }
  
  inline void addData(Integer id, const T data) { 
    m_data[id].add(data);
  }
  
  inline UniqueArray<T>& at(Integer id) {
    return m_data[id];
  }
  
  inline ConstArrayView<T> at(Integer id) const {
    return m_data[id];
  }


private:
  
  UniqueArray< UniqueArray<T> > m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchange::
MeshExchange(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_parallel_mng(mesh->parallelMng())
, m_nb_rank(m_parallel_mng->commSize())
, m_rank(m_parallel_mng->commRank())
, m_cell_family(mesh->itemFamily(IK_Cell))
, m_neighbour_cells_owner(NULL)
, m_neighbour_cells_new_owner(NULL)
, m_neighbour_extra_cells_owner(NULL)
, m_neighbour_extra_cells_new_owner(NULL)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchange::
~MeshExchange()
{
  for( const auto& itosend : m_items_to_send )
    delete itosend.second;
  delete m_neighbour_cells_owner;
  delete m_neighbour_cells_new_owner;
  delete m_neighbour_extra_cells_owner;
  delete m_neighbour_extra_cells_new_owner;
  for( const auto& idestrank : m_item_dest_ranks_map)
    delete idestrank.second;
  for (const auto& ighostdestrank_map : m_ghost_item_dest_ranks_map)
    for (const auto& ighostdestrank : ighostdestrank_map)
      delete ighostdestrank.second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste par sous-domaine des entités à envoyer pour la famille \a family.
ConstArrayView<std::set<Int32>> MeshExchange::
getItemsToSend(IItemFamily* family) const
{
  auto iter = m_items_to_send.find(family);
  if (iter==m_items_to_send.end())
    ARCANE_FATAL("No items to send for family '{0}'",family->name());
  return iter->second->constView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste par sous-domaine des entités à envoyer pour la famille \a family.
ArrayView<std::set<Int32>> MeshExchange::
_getItemsToSend(IItemFamily* family)
{
  auto iter = m_items_to_send.find(family);
  if (iter==m_items_to_send.end())
    ARCANE_FATAL("No items to send for family '{0}'",family->name());
  return iter->second->view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
computeInfos()
{
  // Créé les tableaux contenant pour chaque famille la liste des entités
  // à envoyer.
  for( IItemFamily* family : m_mesh->itemFamilies()){
    m_items_to_send.insert(std::make_pair(family,new UniqueArray< std::set<Int32> >));
  }

  Int32ConstArrayView cells_new_owner(m_cell_family->itemsNewOwner().asArray());
  //! AMR
  if(m_mesh->isAmrActivated()){
    _computeMeshConnectivityInfos2(cells_new_owner);
    _computeGraphConnectivityInfos();
    _exchangeCellDataInfos(cells_new_owner,true);
    _markRemovableCells(cells_new_owner,true);
    _markRemovableParticles();
    _computeItemsToSend2();
  }
  else if (m_mesh->itemFamilyNetwork() && m_mesh->itemFamilyNetwork()->isActivated())
  {
    if(m_mesh->useMeshItemFamilyDependencies())
    {
      _computeMeshConnectivityInfos3();
      _computeGraphConnectivityInfos();
      _exchangeCellDataInfos3(); // todo renommer itemDataInfo
      _exchangeGhostItemDataInfos();
      _markRemovableItems();
      _markRemovableParticles();
      _computeItemsToSend3();
    }
    else
    {
      //Manage Mesh item_families in standard way
      _computeMeshConnectivityInfos(cells_new_owner);
      _computeMeshConnectivityInfos3();
      _computeGraphConnectivityInfos();
      _exchangeCellDataInfos(cells_new_owner,false);
      _exchangeCellDataInfos3();
      //_exchangeGhostItemDataInfos();
      //_markRemovableItems();
      _markRemovableDoFs();
      _markRemovableCells(cells_new_owner,false);
      _markRemovableParticles();
      _computeItemsToSend(true);
    }
  }
  else
  {
    _computeMeshConnectivityInfos(cells_new_owner);
    _computeGraphConnectivityInfos();
    _exchangeCellDataInfos(cells_new_owner,false);
    _markRemovableCells(cells_new_owner,false);
    _markRemovableParticles();
    _computeItemsToSend();
  } //! AMR END
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeGraphConnectivityInfos()
{
  const Integer cell_variable_size = m_cell_family->maxLocalId();
  
  Int32UniqueArray tmp_new_owner(m_nb_rank);
  tmp_new_owner.fill(NULL_ITEM_ID);
  Int32UniqueArray tmp_owner(m_nb_rank);
  tmp_owner.fill(NULL_ITEM_ID);
  
  Int32UniqueArray tmp_link_new_owner(m_nb_rank);
  tmp_link_new_owner.fill(NULL_ITEM_ID);
  Int32UniqueArray tmp_link_owner(m_nb_rank);
  tmp_link_owner.fill(NULL_ITEM_ID);
  
  mesh::NewItemOwnerBuilder owner_builder;
  
  m_neighbour_extra_cells_owner     = new DynamicMultiArray<Int32>(cell_variable_size);
  m_neighbour_extra_cells_new_owner = new DynamicMultiArray<Int32>(cell_variable_size);
  
  if (m_mesh->itemFamilyNetwork() && m_mesh->itemFamilyNetwork()->isActivated())
  {
      _addGraphConnectivityToNewConnectivityInfo();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addGraphConnectivityToNewConnectivityInfo()
{
  ENUMERATE_CELL(icell, m_cell_family->allItems()) {
    Int32Array& extra_new_owners = m_neighbour_extra_cells_new_owner->at(icell.localId());
    Int32Array& extra_owners = m_neighbour_extra_cells_owner->at(icell.localId());
    ItemDestRankArray* item_dest_ranks = nullptr;
    if (icell->isOwn())
      item_dest_ranks = m_item_dest_ranks_map[m_cell_family];
    else
      item_dest_ranks= m_ghost_item_dest_ranks_map[icell->owner()][m_cell_family];
    item_dest_ranks->at(icell.localId()).addRange(extra_new_owners);
    item_dest_ranks->at(icell.localId()).addRange(extra_owners);
  }
  // This exchange modify dest ranks of cells => need to update their dependencies (costly ??)
  _propagatesToChildDependencies(m_cell_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeMeshConnectivityInfos(Int32ConstArrayView cells_new_owner)
{
  Integer cell_variable_size = m_cell_family->maxLocalId();
    
  m_neighbour_cells_owner     = new IncrementalUnorderedMultiArray<Int32>(cell_variable_size);
  m_neighbour_cells_new_owner = new IncrementalUnorderedMultiArray<Int32>(cell_variable_size);
  
  Int32UniqueArray tmp_new_owner(m_nb_rank);
  tmp_new_owner.fill(NULL_ITEM_ID);
  Int32UniqueArray tmp_owner(m_nb_rank);
  tmp_owner.fill(NULL_ITEM_ID);
  
  ENUMERATE_CELL(icell,m_cell_family->allItems().own()){
    Cell cell = *icell;
    Integer cell_local_id = cell.localId();
    // On ne se rajoute pas à notre liste de maille connectée
    tmp_owner[m_rank] = cell_local_id;
    
    m_neighbour_cells_owner->beginIncrement(cell_local_id);
    m_neighbour_cells_new_owner->beginIncrement(cell_local_id);

    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
      for( CellEnumerator icell2((*inode).cells()); icell2.hasNext(); ++icell2 ){
        Integer cell2_local_id = icell2.localId();
        Integer cell2_new_owner = cells_new_owner[cell2_local_id];
        Integer cell2_owner = icell2->owner();
        // Regarde si on n'est pas dans la liste et si on n'y est pas,
        // s'y rajoute
        if (tmp_new_owner[cell2_new_owner]!=cell_local_id){
          tmp_new_owner[cell2_new_owner] = cell_local_id;
          m_neighbour_cells_new_owner->addData(cell2_new_owner);
        }
        if (tmp_owner[cell2_owner]!=cell_local_id){
          tmp_owner[cell2_owner] = cell_local_id;
          m_neighbour_cells_owner->addData(cell2_owner);
        }
      }
    }
    m_neighbour_cells_owner->endIncrement();
    m_neighbour_cells_new_owner->endIncrement();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_exchangeCellDataInfos([[maybe_unused]] Int32ConstArrayView cells_new_owner,bool use_active_cells)
{
  auto sd_exchange { ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  Int32UniqueArray recv_sub_domains;
  m_cell_family->getCommunicatingSubDomains(recv_sub_domains);
  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i )
    sd_exchange->addSender(recv_sub_domains[i]);
  
  sd_exchange->initializeCommunicationsMessages(recv_sub_domains);

  UniqueArray<Int64> cells_to_comm_uid;
  UniqueArray<Int32> cells_to_comm_owner;
  UniqueArray<Int32> cells_to_comm_owner_size;
  UniqueArray<Int32> cells_to_comm_new_owner;
  UniqueArray<Int32> cells_to_comm_new_owner_size;

  ItemGroup own_items = m_cell_family->allItems().own();
  ItemGroup all_items = m_cell_family->allItems();
  // Avec l'AMR, on utilise les mailles actives et pas toutes les mailles.
  if (use_active_cells){
    own_items = m_cell_family->allItems().ownActiveCellGroup();
    all_items = m_cell_family->allItems().activeCellGroup();
  }

  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i ){
    ISerializeMessage* comm = sd_exchange->messageToSend(i);
    Int32 dest_sub_domain = comm->destination().value();
    ISerializer* sbuf = comm->serializer();

    cells_to_comm_uid.clear();
    cells_to_comm_owner.clear();
    cells_to_comm_owner_size.clear();
    cells_to_comm_new_owner.clear();
    cells_to_comm_new_owner_size.clear();

    ENUMERATE_CELL(icell,own_items){
      Cell cell = *icell;
      Integer cell_local_id = cell.localId();
      Int32ConstArrayView owners = m_neighbour_cells_owner->at(cell_local_id);
      bool need_send = owners.contains(dest_sub_domain);
      Int32ConstArrayView extra_owners = m_neighbour_extra_cells_owner->at(cell_local_id);
      if (!need_send){
        need_send = extra_owners.contains(dest_sub_domain);
      }
      if (!need_send)
        continue;

      Int32ConstArrayView new_owners = m_neighbour_cells_new_owner->at(cell_local_id);
      const Integer nb_new_owner = new_owners.size();
      const Integer nb_owner = owners.size();
      
      Int32ConstArrayView extra_new_owner = m_neighbour_extra_cells_new_owner->at(cell_local_id);
      const Integer nb_extra_new_owner = extra_new_owner.size();
      
      cells_to_comm_uid.add(cell.uniqueId().asInt64());
      cells_to_comm_owner_size.add(nb_owner);
      cells_to_comm_new_owner_size.add(nb_new_owner+nb_extra_new_owner);
      for( Integer zz=0; zz<nb_owner; ++zz )
        cells_to_comm_owner.add(owners[zz]);
      for( Integer zz=0; zz<nb_new_owner; ++zz )
        cells_to_comm_new_owner.add(new_owners[zz]); 
      for( Integer zz=0; zz<nb_extra_new_owner; ++zz )
        cells_to_comm_new_owner.add(extra_new_owner[zz]);
    }
    sbuf->setMode(ISerializer::ModeReserve);

    sbuf->reserveInt64(1); // Pour le nombre de mailles
    sbuf->reserveArray(cells_to_comm_uid);
    sbuf->reserveArray(cells_to_comm_owner_size);
    sbuf->reserveArray(cells_to_comm_owner);
    sbuf->reserveArray(cells_to_comm_new_owner_size);
    sbuf->reserveArray(cells_to_comm_new_owner);

    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);

    sbuf->putInt64(cells_to_comm_uid.size());
    sbuf->putArray(cells_to_comm_uid);
    sbuf->putArray(cells_to_comm_owner_size);
    sbuf->putArray(cells_to_comm_owner);
    sbuf->putArray(cells_to_comm_new_owner_size);
    sbuf->putArray(cells_to_comm_new_owner);
  }

  sd_exchange->processExchange();

  Int32UniqueArray cells_to_comm_local_id;
  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i ){
    ISerializeMessage* comm = sd_exchange->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    Integer owner_index = 0;
    Integer new_owner_index = 0;
    Int64 nb_cell = sbuf->getInt64();
    sbuf->getArray(cells_to_comm_uid);
    sbuf->getArray(cells_to_comm_owner_size);
    sbuf->getArray(cells_to_comm_owner);
    sbuf->getArray(cells_to_comm_new_owner_size);
    sbuf->getArray(cells_to_comm_new_owner);
    cells_to_comm_local_id.resize(nb_cell);
    m_cell_family->itemsUniqueIdToLocalId(cells_to_comm_local_id,cells_to_comm_uid);
    for( Integer icell=0; icell<nb_cell; ++icell ){
      Integer cell_local_id = cells_to_comm_local_id[icell];
      Integer cell_nb_owner = cells_to_comm_owner_size[icell];
      Integer cell_nb_new_owner = cells_to_comm_new_owner_size[icell];
#ifdef DEBUG
      info()<< " cell "<<icell
            << " lid=" <<cell_local_id
            << " uid=" <<cells_to_comm_uid[icell]
            << " ind=" <<m_neighbour_cells_owner->index(cell_local_id);
#endif
      if (m_neighbour_cells_owner->index(cell_local_id)!=(-1))
        fatal() << "Cell uid=" << cells_to_comm_uid[icell] << " already has neighbours 'owner'!";
      if (m_neighbour_cells_new_owner->index(cell_local_id)!=(-1))
        fatal() << "Cell uid=" << cells_to_comm_uid[icell] << " already has neighbours 'new_owner'!";
      m_neighbour_cells_owner->beginIncrement(cell_local_id);
      m_neighbour_cells_new_owner->beginIncrement(cell_local_id);
#ifdef DEBUG
      info()<< " cell "<< icell
            << " lid=" << cell_local_id
            << " uid=" << cells_to_comm_uid[icell]
            << " ind=" << m_neighbour_cells_owner->index(cell_local_id);
#endif
      for( Integer zz=0; zz<cell_nb_owner; ++zz )
        m_neighbour_cells_owner->addData(cells_to_comm_owner[owner_index+zz]);
      owner_index += cell_nb_owner;
      m_neighbour_cells_owner->endIncrement();

      for( Integer zz=0; zz<cell_nb_new_owner; ++zz )
        m_neighbour_cells_new_owner->addData(cells_to_comm_new_owner[new_owner_index+zz]);
      new_owner_index += cell_nb_new_owner;
      m_neighbour_cells_new_owner->endIncrement();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addItemToSend(ArrayView< std::set<Int32> > items_to_send,
               Int32 local_id,Int32 cell_local_id,
               bool use_itemfamily_network)
{
  Int32ConstArrayView new_owners = m_neighbour_cells_new_owner->at(cell_local_id);
  for( Integer zz=0, nb_new_owner = new_owners.size(); zz<nb_new_owner; ++zz )
    items_to_send[new_owners[zz]].insert(local_id);

  Int32ConstArrayView extra_new_owners = m_neighbour_extra_cells_new_owner->at(cell_local_id);
  for( Integer zz=0, nb_extra_new_owner = extra_new_owners.size(); zz<nb_extra_new_owner; ++zz )
    items_to_send[extra_new_owners[zz]].insert(local_id);

  if(use_itemfamily_network)
  {
    Int32ConstArrayView network_new_owners = m_item_dest_ranks_map[m_cell_family]->at(cell_local_id);
    for( Integer zz=0, nb_network_new_owner = network_new_owners.size(); zz<nb_network_new_owner; ++zz )
    {
      items_to_send[network_new_owners[zz]].insert(local_id);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeItemsToSend(bool send_dof)
{
  for( const auto& iter : m_items_to_send )
    iter.second->resize(m_nb_rank);

  IItemFamily* node_family = mesh()->nodeFamily();
  IItemFamily* edge_family = mesh()->edgeFamily();
  IItemFamily* face_family = mesh()->faceFamily();
  IItemFamily* cell_family = m_cell_family;

  ArrayView<std::set<Int32>> nodes_to_send = _getItemsToSend(node_family);
  ArrayView<std::set<Int32>> edges_to_send = _getItemsToSend(edge_family);
  ArrayView<std::set<Int32>> faces_to_send = _getItemsToSend(face_family);
  ArrayView<std::set<Int32>> cells_to_send = _getItemsToSend(cell_family);

  Int32ConstArrayView cells_new_owner(cell_family->itemsNewOwner().asArray());
  Int32ConstArrayView faces_new_owner(face_family->itemsNewOwner().asArray());
  bool use_itemfamily_network = m_mesh->itemFamilyNetwork() && m_mesh->itemFamilyNetwork()->isActivated() ;
  ENUMERATE_CELL(icell,cell_family->allItems().own()){
    _addItemToSend(cells_to_send,icell.itemLocalId(),icell.itemLocalId(),use_itemfamily_network);
  }

  ENUMERATE_NODE(inode,node_family->allItems().own()){
    Node node = *inode;
    Integer node_local_id = node.localId();
    for( CellEnumerator icell(node.cells()); icell.hasNext(); ++icell )
      _addItemToSend(nodes_to_send,node_local_id,icell.localId(),use_itemfamily_network);
  }

  ENUMERATE_EDGE(iedge,edge_family->allItems().own()){
    Edge edge = *iedge;
    Integer edge_local_id = edge.localId();
    for( CellEnumerator icell(edge.cells()); icell.hasNext(); ++icell )
      _addItemToSend(edges_to_send,edge_local_id,icell.localId(),use_itemfamily_network);
  }

  ENUMERATE_FACE(iface,face_family->allItems().own()){
    Face face = *iface;
    Integer face_local_id = face.localId();
    for( CellEnumerator icell(face.cells()); icell.hasNext(); ++icell ){
      _addItemToSend(faces_to_send,face_local_id,icell.localId(),use_itemfamily_network);
    }
  }
  
  {
    for( IItemFamily* family : m_mesh->itemFamilies())
    {
      IParticleFamily* particle_family = family->toParticleFamily();
      if (particle_family && particle_family->getEnableGhostItems()==true){
        ArrayView<std::set<Int32>> to_send = _getItemsToSend(family);
        ENUMERATE_PARTICLE(iparticle,particle_family->allItems().own()){
          _addItemToSend(to_send,iparticle->localId(),iparticle->cell().localId(),use_itemfamily_network);
        }
      }
      if(send_dof && family->itemKind()==IK_DoF)
      {
        _setItemsToSend(family);
      }
    }
  }

  // S'assure qu'on ne s'envoie pas les entités
  for( const auto& iter : m_items_to_send )
    (*(iter.second))[m_rank].clear();

  const bool is_print = false;
  if (is_print){
    debug() << "PRINT ITEM TO SEND V1";
    _printItemToSend(node_family);
    _printItemToSend(edge_family);
    _printItemToSend(face_family);
    _printItemToSend(cell_family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette version est plus générale que la première puisque elle prend en compte
 * le cas AMR. En revanche, la version actuelle est moins performante que la première.
 * C'est pour cela, la version initiale restera comme version par defaut dans le cas
 * non AMR.
 * NOTE: la version initiale, peut être utiliser dans le cas amr comme les échanges
 * ce font par rapport à la connectivité autour des noeuds. Donc l'échange se fait
 * sur l'arbre d'une maille active. Cette idée n'a pas été TODO testée pour la confirmer.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeMeshConnectivityInfos2(Int32ConstArrayView cells_new_owner)
{
  Integer cell_variable_size = m_cell_family->maxLocalId();
    
  m_neighbour_cells_owner     = new IncrementalUnorderedMultiArray<Int32>(cell_variable_size);
  m_neighbour_cells_new_owner = new IncrementalUnorderedMultiArray<Int32>(cell_variable_size);
  
  Int32UniqueArray tmp_new_owner(m_nb_rank);
  tmp_new_owner.fill(NULL_ITEM_ID);
  Int32UniqueArray tmp_owner(m_nb_rank);
  tmp_owner.fill(NULL_ITEM_ID);
  
  ENUMERATE_CELL(icell,m_cell_family->allItems().ownActiveCellGroup()){
    const Cell& cell = *icell;
    Integer cell_local_id = cell.localId();
    // On ne se rajoute pas à notre liste de maille connectée
    tmp_owner[m_rank] = cell_local_id;
    
    m_neighbour_cells_owner->beginIncrement(cell_local_id);
    m_neighbour_cells_new_owner->beginIncrement(cell_local_id);

    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
      Int32UniqueArray local_ids;
      for( CellEnumerator icell2((*inode)._internalActiveCells(local_ids)); icell2.hasNext(); ++icell2 ){
        Integer cell2_local_id = icell2.localId();
        Integer cell2_new_owner = cells_new_owner[cell2_local_id];
        Integer cell2_owner = icell2->owner();
        // Regarde si on n'est pas dans la liste et si on n'y est pas,
        // s'y rajoute
        if (tmp_new_owner[cell2_new_owner]!=cell_local_id){
          tmp_new_owner[cell2_new_owner] = cell_local_id;
          m_neighbour_cells_new_owner->addData(cell2_new_owner);
        }
        if (tmp_owner[cell2_owner]!=cell_local_id){
          tmp_owner[cell2_owner] = cell_local_id;
          m_neighbour_cells_owner->addData(cell2_owner);
        }
      }
      local_ids.clear();
    }
    m_neighbour_cells_owner->endIncrement();
    m_neighbour_cells_new_owner->endIncrement();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addTreeCellToSend(ArrayView< std::set<Int32> > items_to_send,
                   Int32 local_id,Int32 cell_local_id,
                   CellInfoListView cells)
{
  Int32ConstArrayView new_owners = m_neighbour_cells_new_owner->at(cell_local_id);
  
	Cell cell(cells[local_id]);
	if(cell.level() == 0){
    for( Integer zz=0, nb_new_owner = new_owners.size(); zz<nb_new_owner; ++zz )
      items_to_send[new_owners[zz]].insert(local_id);
    // Graphe ok sur le niveau actif
    Int32ConstArrayView extra_new_owners = m_neighbour_extra_cells_new_owner->at(cell_local_id);
    for( Integer zz=0, nb_extra_new_owner = extra_new_owners.size(); zz<nb_extra_new_owner; ++zz )
      items_to_send[extra_new_owners[zz]].insert(local_id);
	}
	else {
    Int32UniqueArray family;
    Cell top_parent= cell.topHParent();
    _familyTree(family,top_parent);
    for(Integer c=0,cs=family.size();c<cs;c++){
      for( Integer zz=0, nb_new_owner = new_owners.size(); zz<nb_new_owner; ++zz )
        items_to_send[new_owners[zz]].insert(family[c]);
		}
	} 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addTreeItemToSend(Int32 cell_local_id,CellInfoListView cells)
{
  Int32UniqueArray tree_cells_lid;
  
  Int32ConstArrayView new_owners = m_neighbour_cells_new_owner->at(cell_local_id);
	
  Cell cell(cells[cell_local_id]);

  Int32UniqueArray family;

	Cell top_parent= cell.topHParent();
	_familyTree(family,top_parent);

  IItemFamily* node_family = mesh()->nodeFamily();
  IItemFamily* edge_family = mesh()->edgeFamily();
  IItemFamily* face_family = mesh()->faceFamily();

  ArrayView<std::set<Int32>> nodes_to_send = _getItemsToSend(node_family);
  ArrayView<std::set<Int32>> edges_to_send = _getItemsToSend(edge_family);
  ArrayView<std::set<Int32>> faces_to_send = _getItemsToSend(face_family);

	// On suppose que les containers nodes_to_send, edges_to_send et face_to_send
	// sont deja alloues
	for(Integer c=0,cs=family.size();c<cs;c++) // c=0 est la maille topParent.
		for( Integer zz=0, nb_new_owner = new_owners.size(); zz<nb_new_owner; ++zz ){
			Cell cell2 = cells[family[c]];
			for( Node node : cell2.nodes() )
				nodes_to_send[new_owners[zz]].insert(node.localId());
			for( Edge edge : cell2.edges() )
				edges_to_send[new_owners[zz]].insert(edge.localId());
			for( Face face : cell2.faces() )
				faces_to_send[new_owners[zz]].insert(face.localId());
		}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_familyTree (Int32Array& family,Cell item, const bool reset) const
{
	ARCANE_ASSERT((!item.itemBase().isSubactive()),("The family tree doesn't include subactive items"));
	// Clear the array if the flag reset tells us to.
	if (reset)
		family.clear();
	// Add this item to the family tree.
	family.add(item.localId());
	// Recurse into the items children, if it has them.
	// Do not clear the array any more.
	if (!item.isActive())
		for (Integer c=0, cs=item.nbHChildren(); c<cs; c++){
			Item ichild = item.hChild(c);
			if (ichild.isOwn())
				_familyTree (family,ichild.toCell(), false);
		}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeItemsToSend2()
{
  for( const auto& iter : m_items_to_send )
    iter.second->resize(m_nb_rank);

  IItemFamily* node_family = mesh()->nodeFamily();
  IItemFamily* edge_family = mesh()->edgeFamily();
  IItemFamily* face_family = mesh()->faceFamily();

  ArrayView<std::set<Int32>> nodes_to_send = _getItemsToSend(node_family);
  ArrayView<std::set<Int32>> edges_to_send = _getItemsToSend(edge_family);
  ArrayView<std::set<Int32>> faces_to_send = _getItemsToSend(face_family);
  ArrayView<std::set<Int32>> cells_to_send = _getItemsToSend(m_cell_family);

  Int32ConstArrayView cells_new_owner(m_cell_family->itemsNewOwner().asArray());
  Int32ConstArrayView faces_new_owner(face_family->itemsNewOwner().asArray());

  CellInfoListView cells(m_cell_family);
  ENUMERATE_CELL(icell,m_cell_family->allItems().ownActiveCellGroup()){
	  Cell cell = *icell;
	  _addTreeCellToSend(cells_to_send,icell.itemLocalId(),icell.itemLocalId(),cells);
    
    if(cell.level() == 0){
      
      for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
			  Node node= *inode;
			  if(node.isOwn()){
				  Integer node_local_id = node.localId();
				  for( CellEnumerator icell2(node.cells()); icell2.hasNext(); ++icell2 ){
					  Cell cc= *icell2;
					  if(cc.isActive())
						  _addItemToSend(nodes_to_send,node_local_id,icell2.localId());
				  }
			  }
		  }
      
		  for( EdgeEnumerator iedge(cell.edges()); iedge.hasNext(); ++iedge ){
			  Edge edge = *iedge;
			  if(edge.isOwn()){
				  Integer edge_local_id = edge.localId();
				  for( CellEnumerator icell2(edge.cells()); icell2.hasNext(); ++icell2 )
					  _addItemToSend(edges_to_send,edge_local_id,icell2.localId());
			  }
		  }
      
		  for(FaceEnumerator iface(cell.faces());iface.hasNext();++iface){
			  Face face = *iface;
			  if(face.isOwn()){
				  Integer face_local_id = face.localId();
				  for( CellEnumerator icell2(face.cells()); icell2.hasNext(); ++icell2 ){
					  _addItemToSend(faces_to_send,face_local_id,icell2.localId());
				  }
			  }
		  }
      
    }
    else if(cell.level() > 0)
		  _addTreeItemToSend(icell.itemLocalId(),cells);
  }
 
  for( IItemFamily* family : m_mesh->itemFamilies()){
    IParticleFamily* particle_family = family->toParticleFamily();
    if (particle_family && particle_family->getEnableGhostItems()==true){
      ArrayView<std::set<Int32>> to_send = _getItemsToSend(family);
      ENUMERATE_PARTICLE(iparticle,particle_family->allItems().own()){
        _addItemToSend(to_send,iparticle->localId(),iparticle->cell().localId());
      }
    }
  }
 
  // S'assure qu'on ne s'envoie pas les entités
  for( const auto& iter : m_items_to_send )
    (*(iter.second))[m_rank].clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR OFF

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette version propose une implémentation du calcul des items à échanger
 * utilisant le graphe de dépendances des familles (ItemFamilyNetwork)
 * NOTE: Pour l'instant les algos pour le graphe d'arcane IGraph et pour les familles
 * de particules sont inchangés.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! ITEM_FAMILY_NETWORK ON

void MeshExchange::
_computeMeshConnectivityInfos3()
{
  if (!m_mesh->itemFamilyNetwork()) info() << "Should have an IItemFamilyNetwork. Exiting.";
  //1-Prepare data structure
  m_ghost_item_dest_ranks_map.resize(m_parallel_mng->commSize());
  m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
      _allocData(family);
      },
      IItemFamilyNetwork::TopologicalOrder);

  // Here the algorithm propagates the owner to the neighborhood
  // The propagation to the items owned (dependencies) has already been done in updateOwnersFromCell
  // Todo include also the owned item propagation (would avoid to call updateOwnersFromCell...)
  // What we do here is to build the number of ghost layers needed
  //-For each ghost layer wanted :
  // 1- For all items add Item new owner and all its dest ranks to all child item connected (dependency or relation) dest ranks
  //    This is done for all family, parsing the family graph from head to leaves
  // 2- For all items add new dest ranks to its depending child (move with your downward dependencies)
  if (!m_mesh->ghostLayerMng()) info() << "Should have a IGhostLayerMng. Exiting";
  for (int ghost_layer_index = 0; ghost_layer_index < m_mesh->ghostLayerMng()->nbGhostLayer(); ++ghost_layer_index)
  {
    //2-Diffuse destination rank info to connected items
    m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
      _propagatesToChildConnectivities(family);
    },
    IItemFamilyNetwork::TopologicalOrder);

    //m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
    //  _propagatesToChildDependencies(family);
    //},
    //IItemFamilyNetwork::TopologicalOrder);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_propagatesToChildConnectivities(IItemFamily* family)
{
  VariableItemInt32& item_new_owner = family->itemsNewOwner();
  /* TODO : this propagation should be needed for all child dependencies and not all relations but only ghost providing relations
   * eg for node to cell connectivity if ghosts are found via node neighbouring. This notion of ghost providing relations must be added
   * In traditional case only node to cell is a ghost providing relation => a test can be made to add only this relation (along with all dependencies) :
   * auto child_connectivities = m_mesh->itemFamilyNetwork()->getChildDependencies(family); // instead of getChildConnectivities
   * if (family->itemKind()==IK_Node) {
   * child_connectivities.add(m_mesh->itemFamilyNetwork()->getConnectivity(family,m_mesh->cellFamily(),mesh::connectivityName(family,m_mesh->cellFamily())));
   * }
   */
  //auto child_connectivities = m_mesh->itemFamilyNetwork()->getChildConnectivities(family);
  auto child_connectivities = m_mesh->itemFamilyNetwork()->getChildDependencies(family); // Only dependencies are required to propagate owner
  for (const auto& child_connectivity : child_connectivities){
    if(child_connectivity)
    {
      VariableItemInt32& conn_item_new_owner = child_connectivity->targetFamily()->itemsNewOwner();
      auto accessor = IndexedItemConnectivityAccessor(child_connectivity);
      ENUMERATE_ITEM(item, family->allItems()){
        // Parse child relations
        _addDestRank(*item,family,item_new_owner[item]);
        ENUMERATE_ITEM(connected_item,accessor(ItemLocalId(item))){
          _addDestRank(*item,family,conn_item_new_owner[connected_item]);
        }

        ENUMERATE_ITEM(connected_item,accessor(ItemLocalId(item))){
          _addDestRank(*connected_item,child_connectivity->targetFamily(),*item,family);
        }
      }
    }
  }
  if(!m_mesh->useMeshItemFamilyDependencies()){
    switch(family->itemKind()){
    case IK_Face:
      ENUMERATE_(Face, item, family->allItems()) {
        for( Cell cell : item->cells())
          _addDestRank(cell,m_cell_family,*item,family);
      }
      break;
    case IK_Edge:
      ENUMERATE_(Edge, item, family->allItems()){
        for( Cell cell : item->cells() )
          _addDestRank(cell,m_cell_family,*item,family);
      }
      break;
    case IK_Node:
      ENUMERATE_(Node, item, family->allItems()){
        for( Cell cell : item->cells())
          _addDestRank(cell,m_cell_family,*item,family);
      }
      break;
    default:
      break;
  }
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_propagatesToChildDependencies(IItemFamily* family)
{
  // Here we only propagates dest_ranks, since owner propagation to children has already been done in updateOwnersFromCell
  // todo 1 use these new algo to propagates the owner
  // todo 2 when using the graph we do not need to access to all child connectivities, only the one of the immediately inferior level.
  // => we should allow graph task to take the signature task(IItemFamily*,FirstRankChildren) where FirstRankChildren (better name ?)
  //    would contain the immediate children (FirstRankChildren) would be an EdgeSet or smthg near
  //    => thus we would need to add to the DAG a method children(const Edge&) (see for the name firstRankChildren ??)
  auto child_dependencies =  m_mesh->itemFamilyNetwork()->getChildDependencies(family);
  for (const auto& child_dependency :child_dependencies){
    if(child_dependency)
    {
      auto accessor = IndexedItemConnectivityAccessor(child_dependency);
      ENUMERATE_ITEM(item, family->allItems()){
        // Parse child dependencies
          ENUMERATE_ITEM(connected_item,accessor(ItemLocalId(item))){
            _addDestRank(*connected_item,child_dependency->targetFamily(),*item, family); // as simple as that ??
          }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addDestRank(const Item& item, IItemFamily* item_family, const Integer new_owner) // take an ItemInternal* ?
{
  ItemDestRankArray* item_dest_ranks = nullptr;
  if (item.owner() == m_rank) {
    item_dest_ranks = m_item_dest_ranks_map[item_family];// this search could be written outside the enumerate (but the enumerate is cheap : on connected items)
  }
  else {
    item_dest_ranks= m_ghost_item_dest_ranks_map[item.owner()][item_family];
  }
  // Insert element only if not present
  auto& item_ranks_internal = item_dest_ranks->at(item.localId());
  if (!item_ranks_internal.contains(new_owner))
    item_dest_ranks->at(item.localId()).add(new_owner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_addDestRank(const Item& item, IItemFamily* item_family, const Item& followed_item, IItemFamily* followed_item_family)
{
  // todo : a getDestRank method
  ItemDestRankArray* item_dest_ranks = nullptr;
  if (item.owner() == m_rank)
  {
    item_dest_ranks = m_item_dest_ranks_map[item_family];// this search could be written outside the enumerate (but the enumerate is cheap : on connected items)
  }
  else
  {
    item_dest_ranks= m_ghost_item_dest_ranks_map[item.owner()][item_family];
  }
  ItemDestRankArray* followed_item_dest_ranks = nullptr;
  if (followed_item.owner() == m_rank)
    followed_item_dest_ranks = m_item_dest_ranks_map[followed_item_family];// this search could be written outside the enumerate (but the enumerate is cheap : on connected items)
  else
    followed_item_dest_ranks= m_ghost_item_dest_ranks_map[followed_item.owner()][followed_item_family];
  // Add only new dest rank only if not already present
  auto& new_dest_ranks = followed_item_dest_ranks->at(followed_item.localId());
  auto& current_dest_ranks = item_dest_ranks->at(item.localId());
  IntegerUniqueArray new_dest_rank_to_add;
  new_dest_rank_to_add.reserve((new_dest_ranks.size()));
  for (auto& new_dest_rank : new_dest_ranks){
    if (!current_dest_ranks.contains(new_dest_rank))
      new_dest_rank_to_add.add(new_dest_rank);
  }
  current_dest_ranks.addRange(new_dest_rank_to_add);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_allocData(IItemFamily* family)
{
  m_item_dest_ranks_map[family] = new ItemDestRankArray(family->maxLocalId());
  for (auto& ghost_item_map : m_ghost_item_dest_ranks_map) {
    ghost_item_map[family] = new ItemDestRankArray(family->maxLocalId());
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_debugPrint()
{
  info() << "-ENTERING NEW EXCHANGE DEBUG PRINT";
  for(auto& item_dest_ranks : m_item_dest_ranks_map){
    info() << "--PRINT NEW EXCHANGE INFO";
    info() << "---Destination rank for family " << item_dest_ranks.first->name();
    for (int item_lid = 0; item_lid < item_dest_ranks.first->maxLocalId(); ++item_lid) {
    info() << "---Destination rank for item lid " << item_lid;
      for (auto& item_dest_rank : item_dest_ranks.second->at(item_lid)) {
        info() << "---- Rank  " << item_dest_rank;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_exchangeCellDataInfos3()
{
  // Graph connectivity taken into account thanks to the call to _addGraphConnectivityToNewConnectivityInfo
  auto sd_exchange { ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  Int32UniqueArray recv_sub_domains;
  m_cell_family->getCommunicatingSubDomains(recv_sub_domains);
  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i )
    sd_exchange->addSender(recv_sub_domains[i]);

  sd_exchange->initializeCommunicationsMessages(recv_sub_domains);

  UniqueArray<IItemFamily*> item_families;
  Int32UniqueArray item_lids;
  Int32UniqueArray item_dest_ranks;
  Int32UniqueArray item_nb_dest_ranks;
  Int32UniqueArray family_nb_items;
  Int64UniqueArray item_uids;

  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i ){
      ISerializeMessage* comm = sd_exchange->messageToSend(i);
      Int32 dest_sub_domain = comm->destination().value();
      ISerializer* sbuf = comm->serializer();
      item_families.clear();
      item_dest_ranks.clear();
      item_nb_dest_ranks.clear();
      family_nb_items.clear();
      item_uids.clear();
      for (const auto& family_ghost_item_dest_ranks : m_ghost_item_dest_ranks_map[dest_sub_domain]) {
        Integer family_nb_item = 0;
        item_lids.clear();
        item_families.add(family_ghost_item_dest_ranks.first);
        for (Integer item_lid = 0; item_lid < family_ghost_item_dest_ranks.second->size(); ++item_lid)
        {
          if (family_ghost_item_dest_ranks.second->at(item_lid).size() == 0) continue;
          item_lids.add(item_lid);
          item_dest_ranks.addRange(family_ghost_item_dest_ranks.second->at(item_lid));
          item_nb_dest_ranks.add(family_ghost_item_dest_ranks.second->at(item_lid).size());
          family_nb_item++;
        }
        family_nb_items.add(family_nb_item);
        ENUMERATE_ITEM(item, family_ghost_item_dest_ranks.first->view(item_lids))
        {
          item_uids.add(item->uniqueId().asInt64());
        }
      }
      sbuf->setMode(ISerializer::ModeReserve);
      sbuf->reserveInt64(1); // nb_item_family
      for (const auto& family: item_families)
        sbuf->reserve(family->name()); // ItemFamily->name
      sbuf->reserveInteger(item_families.size()); // ItemFamily->itemKind
      sbuf->reserveArray(item_uids);
      sbuf->reserveArray(family_nb_items);
      sbuf->reserveArray(item_nb_dest_ranks);
      sbuf->reserveArray(item_dest_ranks);

      sbuf->allocateBuffer();
      sbuf->setMode(ISerializer::ModePut);

      sbuf->putInt64(item_families.size());
      for (const auto& family: item_families)
        sbuf->put(family->name()); // ItemFamily->name
      for (const auto& family: item_families)
        sbuf->putInteger(family->itemKind()); // ItemFamily->itemKind
      sbuf->putArray(item_uids);
      sbuf->putArray(family_nb_items);
      sbuf->putArray(item_nb_dest_ranks);
      sbuf->putArray(item_dest_ranks);
  }

  sd_exchange->processExchange();

  for( Integer i=0, n=recv_sub_domains.size(); i<n; ++i ){
    ISerializeMessage* comm = sd_exchange->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    Int64 nb_families = sbuf->getInt64();
    StringUniqueArray item_family_names(nb_families);
    Int32UniqueArray  item_family_kinds(nb_families);
    for (auto& family_name: item_family_names)
      sbuf->get(family_name); // ItemFamily->name
    for (auto& family_kind: item_family_kinds)
      family_kind = sbuf->getInteger(); // ItemFamily->itemKind
    sbuf->getArray(item_uids);
    sbuf->getArray(family_nb_items);
    sbuf->getArray(item_nb_dest_ranks);
    sbuf->getArray(item_dest_ranks);
    Integer item_uid_index = 0;
    Integer item_nb_dest_rank_index = 0;
    Integer item_dest_rank_index = 0;
    for (int family_index = 0; family_index < nb_families; ++family_index)
    {
      IItemFamily* family = m_mesh->findItemFamily(eItemKind(item_family_kinds[family_index]),
                                                   item_family_names[family_index],false);
      Int64ArrayView family_item_uids = item_uids.subView(item_uid_index,family_nb_items[family_index]);
      item_lids.resize(family_item_uids.size());
      family->itemsUniqueIdToLocalId(item_lids,family_item_uids,true);
      for (const auto& item_lid : item_lids){
        auto sub_view = item_dest_ranks.subView(item_dest_rank_index,item_nb_dest_ranks[item_nb_dest_rank_index]);
        m_item_dest_ranks_map[family]->at(item_lid).addRange(sub_view);
        item_dest_rank_index+= item_nb_dest_ranks[item_nb_dest_rank_index++];
      }
      item_uid_index+= family_nb_items[family_index];
    }
  }
  // Dest rank propagation needed since they have been updated for own items
  // copy of propagation part of method computeMeshConnectivityInfo3
  for (int ghost_layer_index = 0; ghost_layer_index < m_mesh->ghostLayerMng()->nbGhostLayer(); ++ghost_layer_index)
  {
    //2-Diffuse destination rank info to connected items
    m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
                                              _propagatesToChildConnectivities(family);
                                          },
                                          IItemFamilyNetwork::TopologicalOrder);

    //m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family){
    //  _propagatesToChildDependencies(family);
    //},
    //IItemFamilyNetwork::TopologicalOrder);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_computeItemsToSend3()
{
  for( const auto& iter : m_items_to_send )
    iter.second->resize(m_nb_rank);

  IItemFamily* node_family = mesh()->nodeFamily();
  IItemFamily* edge_family = mesh()->edgeFamily();
  IItemFamily* face_family = mesh()->faceFamily();
  IItemFamily* cell_family = m_cell_family;

  _setItemsToSend(node_family);
  _setItemsToSend(edge_family);
  _setItemsToSend(face_family);
  _setItemsToSend(cell_family);

  // Tmp for particles : they should be in the family graph
  {
    for( IItemFamily* family : m_mesh->itemFamilies()){
      IParticleFamily* particle_family = family->toParticleFamily();
      if (particle_family && particle_family->getEnableGhostItems()==true){
        ArrayView<std::set<Int32>> to_send = _getItemsToSend(family);
        ENUMERATE_PARTICLE(iparticle,particle_family->allItems().own()){
          for (const auto& dest_rank : m_item_dest_ranks_map[m_cell_family]->at(iparticle->cell().localId())) {
            to_send[dest_rank].insert(iparticle->localId());
          }
        }
      }
    }
  }

  // S'assure qu'on ne s'envoie pas les entités
  for( const auto& iter : m_items_to_send )
    (*(iter.second))[m_rank].clear();

  // SDC DEBUG
//  debug() << "PRINT ITEM TO SEND V2";
//  _printItemToSend(node_family);
//  _printItemToSend(edge_family);
//  _printItemToSend(face_family);
//  _printItemToSend(cell_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_setItemsToSend(IItemFamily* family)
{
  if(family->nbItem()==0)
    return ;
  auto iter = m_items_to_send.find(family);
  if (iter==m_items_to_send.end())
    ARCANE_FATAL("No items to send for family '{0}'",family->name());
  ArrayView<std::set<Int32>> items_to_send = iter->second->view();
  for (Integer item_lid = 0 ; item_lid < m_item_dest_ranks_map[family]->size(); ++item_lid){
    for (const auto& dest_rank : m_item_dest_ranks_map[family]->at(item_lid)){
      items_to_send[dest_rank].insert(item_lid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_printItemToSend(IItemFamily* family)
{
  auto iter = m_items_to_send.find(family);
  if (iter==m_items_to_send.end())
    ARCANE_FATAL("No items to send for family '{0}'",family->name());
  ArrayView<std::set<Int32>> items_to_send = iter->second->view();
  // SDC DEBUG print
  Integer rank = 0;
  debug() << "= ITEM TO SEND FOR FAMILY " << family->name();
  for (const auto& owner_lids : items_to_send) {
    debug() << "== RANK " << rank++;
    for (auto item_lid : owner_lids){
      debug() << "=== has items " << item_lid;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_printItemToRemove(IItemFamily* family)
{
  // SDC DEBUG PRINT
  debug() << "= ITEM TO REMOVE FOR FAMILY " << family->name();
  ENUMERATE_ITEM(item, family->allItems()) {
    if (item->itemBase().flags() & ItemFlags::II_NeedRemove)
      debug() << "== TO REMOVE ITEM " << item->uniqueId()   << " kind " << item->kind();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MeshExchange::
_markRemovableDoFs()
{
  for (const auto& item_dest_ranks_iter : m_item_dest_ranks_map){
    // todo use C++17 structured bindings
    IItemFamily* family = item_dest_ranks_iter.first;
    if(family->itemKind() == IK_DoF ){
      auto & item_new_owners = family->itemsNewOwner();
      ENUMERATE_ITEM(item, family->allItems()){
        Int32ArrayView item_dest_ranks;
        // Get destination rank for item (depending if it's ghost or own)
        if (item->isOwn())
          item_dest_ranks = item_dest_ranks_iter.second->at(item->localId()).view();
        else
          item_dest_ranks = m_ghost_item_dest_ranks_map[item->owner()][family]->at(item->localId()).view();
        // Check if the item must stay on the subdomain (ie dest_rank or new_owner contain the subdomain)
        if (!item_dest_ranks.contains(m_rank) && item_new_owners[item] != m_rank){
          item->mutableItemBase().addFlags(ItemFlags::II_NeedRemove);
        }
      }
    }
//    _printItemToRemove(item_dest_ranks_iter.first); // SDC DEBUG
  }
}

void MeshExchange::
_markRemovableItems(bool with_cell_family)
{
  for (const auto& item_dest_ranks_iter : m_item_dest_ranks_map){
    // todo use C++17 structured bindings
    IItemFamily* family = item_dest_ranks_iter.first;
    if(with_cell_family || family->name()!=m_cell_family->name() ){
      auto & item_new_owners = family->itemsNewOwner();
      ENUMERATE_ITEM(item, family->allItems()){
        Int32ArrayView item_dest_ranks;
        // Get destination rank for item (depending if it's ghost or own)
        if (item->isOwn())
          item_dest_ranks = item_dest_ranks_iter.second->at(item.localId()).view();
        else
          item_dest_ranks = m_ghost_item_dest_ranks_map[item->owner()][family]->at(item.localId()).view();
        // Check if the item must stay on the subdomain (ie dest_rank or new_owner contain the subdomain)
        if (!item_dest_ranks.contains(m_rank) && item_new_owners[item] != m_rank){
          item->mutableItemBase().addFlags(ItemFlags::II_NeedRemove);
        }
      }
    }
    //    _printItemToRemove(item_dest_ranks_iter.first); // SDC DEBUG
  }
}

void MeshExchange::
_markRemovableCells(Int32ConstArrayView cells_new_owner,bool  use_active_cells)
{
  // Ce test n'est plus représentatif avec le concept extraghost
  // Faut il trouver une alternative ??
  // // Vérifie que toutes les mailles ont eues leur voisines calculées
  // ENUMERATE_CELL(icell,m_cell_family->allItems()){
  //   const Cell& cell = *icell;
  //   Integer cell_local_id = cell.localId();
  //   if (m_neighbour_cells_owner->index(cell_local_id) == -1 &&
  //       m_neighbour_extra_cells_owner->at(cell_local_id).size() == 0 )
  //     fatal() << ItemPrinter(cell) << " has no neighbours! (no owner)";
  //   if (m_neighbour_cells_new_owner->index(cell_local_id)==(-1))
  //     fatal() << ItemPrinter(cell) << " has no neighbours! (no new owner index)";
  // }

  // Détermine les mailles qui peuvent être supprimées.
  // Une maille peut-être supprimée si la liste de ses nouveaux propriétaires ne
  // contient pas ce sous-domaine.

  ItemGroup all_items = m_cell_family->allItems();

  auto itemfamily_network = m_mesh->itemFamilyNetwork() ;
  bool use_itemfamily_network = ( itemfamily_network!= nullptr && itemfamily_network->isActivated() );

  // Avec l'AMR, on utilise les mailles actives et pas toutes les mailles.
  if (use_active_cells)
    all_items = m_cell_family->allItems().activeCellGroup();

  ENUMERATE_CELL(icell,all_items){
    Cell cell = *icell;
    Integer cell_local_id = cell.localId();
    if (cells_new_owner[cell_local_id]==m_rank)
      continue;

    Int32ConstArrayView new_owners = m_neighbour_cells_new_owner->at(cell_local_id);
    bool keep_cell = new_owners.contains(m_rank);
    if (!keep_cell){
      Int32ConstArrayView extra_new_owners = m_neighbour_extra_cells_new_owner->at(cell_local_id);
      keep_cell = extra_new_owners.contains(m_rank);
    }
    if(!keep_cell && use_itemfamily_network)
    {
      Int32ArrayView item_dest_ranks;
      // Get destination rank for item (depending if it's ghost or own)
      if (icell->isOwn())
        item_dest_ranks = m_item_dest_ranks_map[m_cell_family]->at(icell->localId()).view();
      else
        item_dest_ranks = m_ghost_item_dest_ranks_map[icell->owner()][m_cell_family]->at(icell->localId()).view();
      // Check if the item must stay on the subdomain (ie dest_rank or new_owner contain the subdomain)
      keep_cell = item_dest_ranks.contains(m_rank) ;
    }
    if (!keep_cell)
    {
      cell.mutableItemBase().addFlags(ItemFlags::II_NeedRemove);
    }
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_markRemovableParticles()
{
  for( IItemFamily* family : m_mesh->itemFamilies()){
    IParticleFamily* particle_family = family->toParticleFamily();
    if(particle_family && particle_family->getEnableGhostItems()==true){
      mesh::NewItemOwnerBuilder owner_builder;

      ENUMERATE_PARTICLE(iparticle,particle_family->allItems()){
        Particle particle = *iparticle;
        Cell cell = owner_builder.connectedCellOfItem(particle);
        particle.mutableItemBase().setFlags(cell.itemBase().flags());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchange::
_checkSubItemsDestRanks()
{
  // Check subitems dest_ranks contain parent dest_rank (even after exchangeDataInfo...)
  // Check for items that the subdomain do send : ie own items
  m_mesh->itemFamilyNetwork()->schedule([this](IItemFamily* family){
    ENUMERATE_ITEM(item, family->allItems().own()) {
      const auto& item_dest_ranks = m_item_dest_ranks_map[family]->at(item.localId());
      for (const auto& child_dependency : m_mesh->itemFamilyNetwork()->getChildDependencies(family))
      {
        auto accessor = IndexedItemConnectivityAccessor(child_dependency);
        ENUMERATE_ITEM(connected_item,accessor(ItemLocalId(item))){
          Int32ConstArrayView subitem_dest_ranks;
          // test can only be done for own subitems, otherwise their dest_ranks are only partially known => to see
          if (connected_item->isOwn()) {
            subitem_dest_ranks = m_item_dest_ranks_map[child_dependency->targetFamily()]->at(connected_item.localId()).constView();
            for (auto dest_rank : item_dest_ranks) {
              if (! subitem_dest_ranks.contains(dest_rank))
                fatal() << "Dest Rank " << dest_rank << " for item " << item->kind() << " uid "<< item->uniqueId()
                        << " not present in subitem " << connected_item->kind() << " uid " << connected_item->uniqueId()
                        << " dest ranks " << subitem_dest_ranks << " ower " << connected_item->owner();
            }
          }
        }
      }
    }
  }, IItemFamilyNetwork::TopologicalOrder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void MeshExchange::
_exchangeGhostItemDataInfos()
{
  /* necessary for dark corner cases (with graph for example) :
   * we need to know if a ghost item will stay ghost
   * otherwise we will destroy it and when we receive it again
   * we won't be able to restore some relations pointing on it
   * ex a cell ghost on Sd_0(subdomain 0) is removed while a face (owner = 0) stays on the subdomain
   * when the cell comes back, send by Sd_1, the face is not send by Sd_1 since it belongs to Sd_0
   * the relation from the cell to the face is restored but the one from the face to the cell is not
   * This case should not happen with contiguous partitions,
   * but when you start to add extra ghost to follow graph for example, it occurs...
   */

  // Graph connectivity taken into account thanks to the call to _addGraphConnectivityToNewConnectivityInfo
  auto sd_exchange { ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  Int32UniqueArray recv_sub_domains;
  m_cell_family->getCommunicatingSubDomains(recv_sub_domains);
  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i )
    sd_exchange->addSender(recv_sub_domains[i]);

  sd_exchange->initializeCommunicationsMessages(recv_sub_domains);

  UniqueArray<IItemFamily*> item_families;
  Int32UniqueArray item_dest_ranks;
  Int32UniqueArray item_nb_dest_ranks;
  Int32UniqueArray family_nb_items;
  Int64UniqueArray item_uids;

  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i ){
    ISerializeMessage* comm = sd_exchange->messageToSend(i);
    Int32 dest_sub_domain = comm->destination().value();
    ISerializer* sbuf = comm->serializer();
    item_families.clear();
    item_dest_ranks.clear();
    item_nb_dest_ranks.clear();
    family_nb_items.clear();
    item_uids.clear();
    for (const auto& family_item_dest_ranks : m_item_dest_ranks_map){
      Integer family_nb_item = 0;
      IItemFamily* family = family_item_dest_ranks.first;
      if (family->nbItem() == 0) continue; // skip empty family
      item_families.add(family);
      // Get shared items with dest_sub_domain
      auto subdomain_index = _getSubdomainIndexInCommunicatingRanks(dest_sub_domain, family->allItemsSynchronizer()->communicatingRanks());
      auto shared_item_lids = family->allItemsSynchronizer()->sharedItems(subdomain_index);
      for (const auto& item_lid : shared_item_lids){
        if (family_item_dest_ranks.second->at(item_lid).size() == 0) continue;
        item_dest_ranks.addRange(family_item_dest_ranks.second->at(item_lid));
        item_nb_dest_ranks.add(family_item_dest_ranks.second->at(item_lid).size());
        family_nb_item++;
      }
      family_nb_items.add(family_nb_item);
      ENUMERATE_ITEM(item, family_item_dest_ranks.first->view(shared_item_lids)) {
        item_uids.add(item->uniqueId().asInt64());
      }
    }
    sbuf->setMode(ISerializer::ModeReserve);

    sbuf->reserveInt64(1); // nb_item_family
    for (const auto& family: item_families)
      sbuf->reserve(family->name()); // ItemFamily->name
    sbuf->reserveInt64(item_families.size()); // ItemFamily->itemKind

    sbuf->reserveArray(item_uids);
    sbuf->reserveArray(family_nb_items);
    sbuf->reserveArray(item_nb_dest_ranks);
    sbuf->reserveArray(item_dest_ranks);

    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);

    sbuf->putInt64(item_families.size());
    for (const auto& family: item_families)
      sbuf->put(family->name()); // ItemFamily->name
    for (const auto& family: item_families)
      sbuf->putInteger(family->itemKind()); // ItemFamily->itemKind

    sbuf->putArray(item_uids);
    sbuf->putArray(family_nb_items);
    sbuf->putArray(item_nb_dest_ranks);
    sbuf->putArray(item_dest_ranks);
  }

  sd_exchange->processExchange();

  for( Integer i=0, is=recv_sub_domains.size(); i<is; ++i ){
    ISerializeMessage* comm = sd_exchange->messageToReceive(i);
    ISerializer* sbuf = comm->serializer();
    Int64 nb_families = sbuf->getInt64();
    StringUniqueArray item_family_names(nb_families);
    Int32UniqueArray  item_family_kinds(nb_families);
    for (auto& family_name: item_family_names)
      sbuf->get(family_name); // ItemFamily->name
    for (auto& family_kind: item_family_kinds)
      family_kind = sbuf->getInteger(); // ItemFamily->itemKind
    sbuf->getArray(item_uids);
    sbuf->getArray(family_nb_items);
    sbuf->getArray(item_nb_dest_ranks);
    sbuf->getArray(item_dest_ranks);
    Integer item_uid_index = 0;
    Integer item_nb_dest_rank_index = 0;
    Integer item_dest_rank_index = 0;
    for (int family_index = 0; family_index < nb_families; ++family_index) {
      IItemFamily* family = m_mesh->findItemFamily(eItemKind(item_family_kinds[family_index]),item_family_names[family_index],false);
      Int64ArrayView family_item_uids = item_uids.subView(item_uid_index,family_nb_items[family_index]);
      Int32UniqueArray item_lids(family_item_uids.size());
      family->itemsUniqueIdToLocalId(item_lids,family_item_uids,true);
      for (const auto& item_lid : item_lids) {
        auto sub_view = item_dest_ranks.subView(item_dest_rank_index,item_nb_dest_ranks[item_nb_dest_rank_index]);
        Int32 dest_rank = comm->destination().value();
        m_ghost_item_dest_ranks_map[dest_rank][family]->at(item_lid).addRange(sub_view);
        item_dest_rank_index+= item_nb_dest_ranks[item_nb_dest_rank_index++];
      }
      item_uid_index+= family_nb_items[family_index];
    }
  }
//  _checkSubItemsDestRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshExchange::
_getSubdomainIndexInCommunicatingRanks(Integer rank, Int32ConstArrayView communicating_ranks)
{
  Integer i = 0;
  while (communicating_ranks[i]!=rank) {
    ++i;
  }
  return i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* MeshExchange::
subDomain() const
{
  return m_mesh->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
