// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityGhostPolicy.h                              (C) 2000-2015 */
/*                                                                           */
/* Interface for the policy of creating ghosts for a connectivity.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMCONNECTIVITYGHOSTPOLICY_H
#define ARCANE_IITEMCONNECTIVITYGHOSTPOLICY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemConnectivityGhostPolicy
{
 public:

  /** Class destructor */
  virtual ~IItemConnectivityGhostPolicy() {}

 public:

  //! Ranks of sub-domains with which we communicate
  virtual Int32ConstArrayView communicatingRanks() = 0;

  /*!
   * \brief gives the local_ids of the items of the family \a family_name shared
   * for the connectivity with the rank process \a rank.
   *
   * The items are repeated as many times as they appear in a connectivity.
   * If they are connected to three items, they appear three times.
   * This array constitutes, with the sharedItemsConnectedItems array which is of the same size,
   * a set of connected item pairs (one element of the connection in each array).
   *
   */
  virtual Int32ConstArrayView sharedItems(const Integer rank, const String& family_name) = 0;

  /*!
   * \brief gives the local_ids of the items of the family \a family_name connected with the sharedItems(rank).
   *
   * The items are repeated as many times as they appear in a connectivity.
   * If they are connected to three items, they appear three times.
   * This array constitutes, with the sharedItems array which is of the same size,
   * a set of connected item pairs (one element of the connection in each array).
   */
  virtual Int32ConstArrayView sharedItemsConnectedItems(const Integer rank, const String& family_name) = 0;

  /*!
   * \brief updates the connectivity by connecting the added ghost items.
   *
   * The two arrays are of the same size, equal to the number of existing 1-to-1 connections.
   * For each connection, the items are repeated, for example for two connections
   * between identifiers 1-3 and 1-4 the arrays will be [ 1 1 ] and [ 3 4 ]
   *
   * This format is identical to that of the two arrays shared_items and shared_items_connected_items.
   * Warning, \a ghost_items_connected_items are unique identifiers (uids). They are the items
   * to which the ghost_items were connected on their owning sub-domain. These items
   * do not necessarily exist on the current sub-domain.
   *
   * TODO: check if it is necessary to provide the families?
   */
  virtual void updateConnectivity(Int32ConstArrayView ghost_items,
                                  Int64ConstArrayView ghost_items_connected_items) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IITEMCONNECTIVITYGHOSTPOLICY_H */
