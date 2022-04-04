// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityGhostPolicy.h                              (C) 2000-2015 */
/*                                                                           */
/* Interface de la politique de création des fantômes pour une connectivité. */
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

  /** Destructeur de la classe */
  virtual ~IItemConnectivityGhostPolicy() {}

 public:

  //! Rangs des sous-domaines avec lesquels on communique
  virtual Int32ConstArrayView communicatingRanks() = 0;

  /*!
   * \brief donne les local_ids des items de la famille \a family_name partagés
   * pour la connectivité avec le processus de rang \a rank.
   *
   * Les items sont répétés autant de fois qu'ils apparaissent dans une connectivité.
   * Ie s'ils sont connectés à trois items, ils apparaissent trois fois.
   *   Ce tableau constitue,avec le tableau sharedItemsConnectedItems qui est de même taille,
   *   un ensemble de pairs d'items connectés (un élément de la connexion dans chaque tableau).
   *
   */
  virtual Int32ConstArrayView sharedItems(const Integer rank, const String& family_name) = 0;

  /*!
   * \brief donne les local_ids des items de la famille \a family_name connectés avec les sharedItems(rank).
   *
   * Les items sont répétés autant de fois qu'ils apparaissent dans une connectivité.
   * Ie s'ils sont connectés à trois items, ils apparaissent trois fois.
   *   Ce tableau constitue,avec le tableau sharedItems qui est de même taille,
   *   un ensemble de pairs d'items connectés (un élément de la connexion dans chaque tableau).
   */
  virtual Int32ConstArrayView sharedItemsConnectedItems(const Integer rank, const String& family_name) = 0;

  /*!
   * \brief mets à jour la connectivité en connectant les items fantômes ajoutés..
   *
   * Les deux tableaux sont de même taille, égale au nombre de connexions 1 pour 1 existantes.
   * Pour chaque connexion, les items sont répétés, par exemple pour deux connexions
   * entre des identifiants 1-3 et 1-4 les tableaux seront [ 1 1 ] et [ 3 4 ]
   *
   * Ce format est identique à celui des deux tableaux shared_items et shared_items_connected_items.
   * Attention, \a ghost_items_connected_items sont des identifiants uniques (uids). Ce sont les items
   * auxquels étaient connectés les ghost_items sur leur sous domaine propriétaire. Ces items
   * n'existent pas forcément sur le sous-domaine courant.
   *
   * TODO: regarder si besoin de donner les familles ?
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
