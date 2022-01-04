// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*------------------------------------------------------------------------------*/
/* GhostLayerFromConnectivityComputer.h                           (C) 2000-2015 */
/*                                                                              */
/* Implémentation d'une politique de création de fantômes pour une connectivité */
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
 * \brief Outil de calcul de la couche fantôme d'une famille à partir
 * de la connectivité.
 *
 * Implémente l'interface IConnectivityGhostPolicy.
 *
 * Cette implémentation définit une politique où les items de la famille "To" seront définis
 * comme partagés s'ils sont connectés à des items de la famille "From" eux-mêmes partagés.
 *
 * On obtient ces items partagés (avec le processus de rang k) de la
 * famille "To" viaa sharedItems(k, ToFamilyName).
 * On obtient les items de la famille "From" auxquels ils sont connectés
 * via sharedItemsConnectedItems(k, FromFamilyName)
 *
 * Le calcul de ces items partagés est fait à la construction de l'objet et
 * à chaque appel de la méthode sharedItems(), pour prendre en compte un éventuelle
 * évolution de la famille "From". Attention donc si la famille a évolué, il faut
 * appeler sharedItems() avant sharedItemsConnectedItems().
 *
 */
class ARCANE_MESH_EXPORT GhostLayerFromConnectivityComputer
: public IItemConnectivityGhostPolicy
{
public:

  /** Constructeur de la classe */
  GhostLayerFromConnectivityComputer(IItemConnectivity* item_to_dofs);

  GhostLayerFromConnectivityComputer() : m_connectivity(NULL), m_trace_mng(NULL) {}

  /** Destructeur de la classe */
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
