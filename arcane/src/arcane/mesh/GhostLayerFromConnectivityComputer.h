// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*------------------------------------------------------------------------------*/
/* GhostLayerFromConnectivityComputer.h                           (C) 2000-2015 */
/*                                                                              */
/* Implementation of a ghost layer creation policy for connectivity             */
/*------------------------------------------------------------------------------*/
#ifndef ARCANE_DOF_GHOSTLAYERFROMCONNECTIVITYCOMPUTER_H
#define ARCANE_DOF_GHOSTLAYERFROMCONNECTIVITYCOMPUTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/IExtraGhostItemsBuilder.h"
#include "arcane/IItemConnectivity.h"

#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/IItemConnectivityGhostPolicy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tool for calculating the ghost layer of a family based on connectivity.
 *
 * Implements the IConnectivityGhostPolicy interface.
 *
 * This implementation defines a policy where items in the "To" family are defined
 * as shared if they are connected to shared items in the "From" family.
 *
 * These shared items (with the rank k process) from the "To" family are obtained via sharedItems(k, ToFamilyName).
 * The items in the "From" family to which they are connected are obtained via sharedItemsConnectedItems(k, FromFamilyName).
 *
 * The calculation of these shared items is done during object construction and
 * upon every call to the sharedItems() method, to account for a possible
 * evolution of the "From" family. Therefore, be careful: if the family has evolved, sharedItems() must be called before sharedItemsConnectedItems().
 *
 */
class ARCANE_MESH_EXPORT GhostLayerFromConnectivityComputer
: public IItemConnectivityGhostPolicy
{
public:

  /** Class constructor */
  GhostLayerFromConnectivityComputer(IItemConnectivity* item_to_dofs);

  GhostLayerFromConnectivityComputer() : m_connectivity(NULL), m_trace_mng(NULL) {}

  /** Class destructor */
  virtual ~GhostLayerFromConnectivityComputer() {}

public:

  //! Interface IItemConnectivityGhostPolicy
  Int32ConstArrayView communicatingRanks();
  Int32ConstArrayView sharedItems(const Integer rank, const String& family_name);
  Int32ConstArrayView sharedItemsConnectedItems(const Integer rank, const String& family_name);
  void updateConnectivity(Int32ConstArrayView ghost_items, Int64ConstArrayView ghost_items_connected_items);


protected:
  IItemConnectivity* m_connectivity;
  ITraceMng* m_trace_mng;
  SharedArray<Int32SharedArray> m_shared_items;
  SharedArray<Int32SharedArray> m_shared_items_connected_items;

private:

  void _computeSharedItems();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
