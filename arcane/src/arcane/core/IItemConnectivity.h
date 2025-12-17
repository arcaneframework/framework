// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivity.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface de connectivité des entités.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITY_H
#define ARCANE_CORE_IITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer une connectivité.
 *
 * Une connectivité relie deux familles, une source (sourceFamily()) et
 * une cible (targetFamily()).
 *
 * Pour récupérer les entités cibles connectées à une entité source, il faut
 * utiliser la classe ConnectivityItemVector. Par exemple:
 *
 * \code
 * IItemConnectivity* c = ...;
 * Item my_item;
 * ConnectivityItemVector civ(c);
 * ENUMERATE_ITEM(icitem,civ.connectedItems(my_item)){
 *  // Itère sur les entités connectées à \a my_item via \a c.
 * }
 * \endcode
 */
class ARCANE_CORE_EXPORT IItemConnectivity
: public IItemConnectivityAccessor
{
 public:

  friend class ConnectivityItemVector;

 public:

  //! Nom de la connectivité
  virtual const String& name() const = 0;

  //! Liste des familles (sourceFamily() + targetFamily())
  virtual ConstArrayView<IItemFamily*> families() const = 0;

  //! Famille source
  virtual IItemFamily* sourceFamily() const = 0;

  //! Famille cible
  virtual IItemFamily* targetFamily() const = 0;

  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) = 0;

  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) = 0;

  /*!
   * \brief Mise à jour de la connectivité.
   *
   * Les deux tableaux \a from_items et \a to_items sont de mêmes tailles.
   * Les items peuvent donc éventuellement être répétés s'ils apparaissent dans
   * plusieurs connexions. Ex si on a les connexions suivantes (en lid) 1-3 ; 2-4 ; 1-5 on entre
   * from_items [ 1 2 1 ] et to_items [ 3 4 5 ].
   */
  virtual void updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
