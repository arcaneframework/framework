// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivitySynchronizer.h                              (C) 2000-2021 */
/*                                                                           */
/* Synchronization des connectivités.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOF_CONNECTIVITYSYNCHRONIZER_H
#define ARCANE_DOF_CONNECTIVITYSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemConnectivity.h"
#include "arcane/IItemConnectivitySynchronizer.h"

#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ExtraGhostItemsManager.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemConnectivityGhostPolicy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemConnectivitySynchronizer
: public IItemConnectivitySynchronizer
, public IExtraGhostItemsBuilder
, public mesh::IExtraGhostItemsAdder
{
 public:

  /** Constructeur de la classe */
  ItemConnectivitySynchronizer(IItemConnectivity* connectivity,
                               IItemConnectivityGhostPolicy* ghost_policy);

 public:

  /*---------------------------------------------------------------------------*/
  //! Interface IConnectivitySynchronizer

  /*!
   * Ajoute le items fantôme définis par IItemConnectivityGhostPolicy.
   * Les uids des items fantômes ajoutés sont conservés.
   * Lors d'un deuxième appel, les fantômes déjà ajoutés ne le sont pas une deuxième fois.
   */

  void synchronize();
  IItemConnectivity* getConnectivity() {return m_connectivity;}

  /*---------------------------------------------------------------------------*/
  //! Interface IExtraGhostItemsBuilder

  void computeExtraItemsToSend();
  IntegerConstArrayView extraItemsToSend(Integer sid) const { return m_data_to_send[sid];}

  /*---------------------------------------------------------------------------*/
  //! Interface IExtraGhostItemsAdder :  add extra ghost in TargetFamily

  void serializeGhostItems(ISerializer* buffer,Int32ConstArrayView ghost_item_lids);
  void addExtraGhostItems (ISerializer* buffer);
  void updateSynchronizationInfo(){ m_connectivity->targetFamily()->computeSynchronizeInfos(); }
  [[deprecated("Y2021: Do not use this method. Try to get 'ISubDomain' from another way")]]
  ISubDomain* subDomain();
  IItemFamily* itemFamily() { return m_connectivity->targetFamily(); }

 private:

  IItemConnectivity* m_connectivity;
  IItemConnectivityGhostPolicy* m_ghost_policy;
  IParallelMng* m_parallel_mng;
  SharedArray<Int32SharedArray> m_data_to_send;
  SharedArray<std::set<Int64> > m_added_ghost;

  void _removeDuplicatedValues(Int64SharedArray& shared_item_uids,
                               IntegerSharedArray& owners);
  void _getItemToSend(Int32SharedArray& shared_items,
                      Int32SharedArray& shared_items_connected_items,
                      const Integer rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* CONNECTIVITYSYNCHRONIZER_H */
