// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivitySynchronizer.h                                  (C) 2000-2015 */
/*                                                                           */
/* Synchronization des connectivités.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOF_CONNECTIVITYSYNCHRONIZER_H
#define ARCANE_DOF_CONNECTIVITYSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISubDomain.h"

#include "arcane/IItemConnectivity.h"
#include "arcane/IItemConnectivitySynchronizer.h"

#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ExtraGhostItemsManager.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
                           IItemConnectivityGhostPolicy* ghost_policy)
    : m_connectivity(connectivity)
    , m_ghost_policy(ghost_policy)
    , m_subdomain(m_connectivity->targetFamily()->subDomain())
    , m_added_ghost(m_subdomain->parallelMng()->commSize()){}

  /** Destructeur de la classe */
  virtual ~ItemConnectivitySynchronizer() {}

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
  ISubDomain* subDomain() { return m_connectivity->targetFamily()->mesh()->subDomain(); }
  IItemFamily* itemFamily() { return m_connectivity->targetFamily(); }

 private:

  IItemConnectivity* m_connectivity;
  IItemConnectivityGhostPolicy* m_ghost_policy;
  ISubDomain* m_subdomain;
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* CONNECTIVITYSYNCHRONIZER_H */
