// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityAccessor.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface des accesseurs des connectivité des entités.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITYACCESSOR_H
#define ARCANE_CORE_IITEMCONNECTIVITYACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer l'accès à une connectivité.
 */
class ARCANE_CORE_EXPORT IItemConnectivityAccessor
{
  friend ConnectivityItemVector;

 public:

  virtual ~IItemConnectivityAccessor() = default;

 public:

  //! Nombre d'entité connectées à l'entité source de numéro local \a lid
  virtual Integer nbConnectedItem(ItemLocalId lid) const = 0;

  //! localId() de la \a index-ième entitée connectées à l'entité source de numéro local \a lid
  virtual Int32 connectedItemLocalId(ItemLocalId lid, Integer index) const = 0;

 protected:

  //! Implémente l'initialisation de \a civ pour cette connectivitée.
  virtual void _initializeStorage(ConnectivityItemVector* civ) = 0;

  //! Remplit \a con_items avec les entités connectées à \a item.
  virtual ItemVectorView _connectedItems(ItemLocalId item, ConnectivityItemVector& con_items) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
