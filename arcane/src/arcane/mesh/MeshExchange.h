// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchange.h                                              (C) 2000-2025 */
/*                                                                           */
/* Echange des entités de maillages entre sous-domaines.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHEXCHANGE_H
#define ARCANE_MESH_MESHEXCHANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

#include "arcane/mesh/MeshGlobal.h"

#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
class IParallelMng;
class IItemFamily;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Echange des entités de maillages entre entre sous-domaines.
 */
class MeshExchange
: public TraceAccessor
{
 private:
  
  template<typename T>
  class IncrementalUnorderedMultiArray;

  template<typename T>
  class DynamicMultiArray;

 public:

  MeshExchange(IMesh* mesh);
  ~MeshExchange();

 public:

  //! Calcule les infos
  void computeInfos();

 public:
 
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get ISubDomain from another way")
  ISubDomain* subDomain() const;

  IMesh* mesh() const { return m_mesh; }

  //! Liste par sous-domaine des entités à envoyer pour la famille \a family.
  ConstArrayView<std::set<Int32>> getItemsToSend(IItemFamily* family) const;

 private:
  
  IMesh* m_mesh; //!< Maillage
  IParallelMng* m_parallel_mng;
  Int32 m_nb_rank;
  Int32 m_rank;
  IItemFamily* m_cell_family;
  
  //! AMR
  void _computeMeshConnectivityInfos2(Int32ConstArrayView cells_new_owner);
  void _addTreeCellToSend(ArrayView< std::set<Int32> > items_to_send,
                          Int32 local_id,Int32 cell_local_id,CellInfoListView cells);
  void _addTreeItemToSend(Int32 cell_local_id, CellInfoListView cells);
  void _addItemToSend2(ArrayView< std::set<Int32> > items_to_send,
                      Int32 item_local_id,Int32 cell_local_id); 
  void _familyTree (Int32Array& family,Cell item, const bool reset=true) const;
  void _computeItemsToSend2();
  
  void _computeMeshConnectivityInfos(Int32ConstArrayView cells_new_owner);
  void _computeGraphConnectivityInfos();
  void _exchangeCellDataInfos(Int32ConstArrayView cells_new_owner,bool use_active_cells);
  void _computeItemsToSend(bool send_dof=false);
  void _addItemToSend(ArrayView< std::set<Int32> > items_to_send,
                      Int32 item_local_id,Int32 cell_local_id,
                      bool use_itemfamily_network=false);

  // Version based on ItemFamilyNetwork
  void _computeMeshConnectivityInfos3();
  void _exchangeCellDataInfos3();
  void _computeItemsToSend3();
  //void _addItemToSend3(ArrayView< std::set<Int32> > items_to_send,
  //                     Int32 item_local_id,Int32 cell_local_id){}
  void _propagatesToChildConnectivities(IItemFamily* family);
  void _propagatesToChildDependencies(IItemFamily* family);
  void _addDestRank(const Item& item, IItemFamily* item_family, const Integer new_owner);
  void _addDestRank(const Item& item, IItemFamily* item_family, const Item& followed_item,
                    IItemFamily* followed_item_family);// add all the followed items destination rank to item
  void _allocData(IItemFamily* family);
  void _addGraphConnectivityToNewConnectivityInfo();

 public:

  std::map< IItemFamily*, UniqueArray< std::set<Int32> >* > m_items_to_send;

 private:

  IncrementalUnorderedMultiArray<Int32>* m_neighbour_cells_owner;
  IncrementalUnorderedMultiArray<Int32>* m_neighbour_cells_new_owner;
  DynamicMultiArray<Int32>* m_neighbour_extra_cells_owner;
  DynamicMultiArray<Int32>* m_neighbour_extra_cells_new_owner;

  //! Liste par sous-domaine des entités à envoyer pour la famille \a family.
  ArrayView<std::set<Int32>> _getItemsToSend(IItemFamily* family);
  void _setItemsToSend(IItemFamily* family);//! Utilisant ItemFamilyNetwork
  void _printItemToSend(IItemFamily* family);// Debug print SDC
  void _printItemToRemove(IItemFamily* family);// Debug print SDC

  void _markRemovableItems(bool with_cell_family=true);
  void _markRemovableDoFs();
  void _markRemovableParticles();
  void _markRemovableCells(Int32ConstArrayView cells_new_owner,bool use_active_cells);

  //  using ItemDestRankArray = IncrementalUnorderedMultiArray<Int32>; // l'un ou l'autre ?
  using ItemDestRankArray = DynamicMultiArray<Int32>;
  using ItemDestRankMap = std::map<IItemFamily*,ItemDestRankArray*>;
  using ItemDestRankMapArray = UniqueArray<ItemDestRankMap>;
  ItemDestRankMap m_item_dest_ranks_map; // [family][item] = [item_destination_ranks]
  ItemDestRankMapArray m_ghost_item_dest_ranks_map; //[rank][family][item]

  void _debugPrint();

  // Check subitems dest_ranks contain parent dest_rank (even after exchangeDataInfo...)
  void _checkSubItemsDestRanks();
  void _exchangeGhostItemDataInfos();
  Integer _getSubdomainIndexInCommunicatingRanks(Integer rank, Int32ConstArrayView communicating_ranks);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
