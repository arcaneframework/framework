// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityMng.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire de connectivité des entités.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITYMNG_H
#define ARCANE_CORE_IITEMCONNECTIVITYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IItemConnectivityMng
{
 public:

  /** Destructeur de la classe */
  virtual ~IItemConnectivityMng() = default;

 public:

  //! Enregistrement d'une connectivité
  virtual void registerConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void unregisterConnectivity(IItemConnectivity* connectivity) = 0;

  virtual void registerConnectivity(IIncrementalItemConnectivity* connectivity) = 0;
  virtual void unregisterConnectivity(IIncrementalItemConnectivity* connectivity) = 0;

  /*!
   * \brief Création d'un objet de synchronisation pour une connectivité.
   *
   * Si la méthode a déjà été appelée pour cette connectivité,
   * un nouveau synchroniseur est créé et le précedent est détruit.
   */
  virtual IItemConnectivitySynchronizer* createSynchronizer(IItemConnectivity* connectivity,
                                                            IItemConnectivityGhostPolicy* connectivity_ghost_policy) = 0;
  virtual IItemConnectivitySynchronizer* getSynchronizer(IItemConnectivity* connectivity) = 0;

  //! Enregistrement de modifications d'une famille d'items
  virtual void setModifiedItems(IItemFamily* family,
                                Int32ConstArrayView added_items,
                                Int32ConstArrayView removed_items) = 0;

  //! Récupération des items modifiés pour mettre à jour une connectivité
  virtual void getSourceFamilyModifiedItems(IItemConnectivity* connectivity,
                                            Int32ArrayView& added_items,
                                            Int32ArrayView& removed_items) = 0;
  virtual void getTargetFamilyModifiedItems(IItemConnectivity* connectivity,
                                            Int32ArrayView& added_items,
                                            Int32ArrayView& removed_items) = 0;

  virtual void getSourceFamilyModifiedItems(IIncrementalItemConnectivity* connectivity,
                                            Int32ArrayView& added_items,
                                            Int32ArrayView& removed_items) = 0;
  virtual void getTargetFamilyModifiedItems(IIncrementalItemConnectivity* connectivity,
                                            Int32ArrayView& added_items,
                                            Int32ArrayView& removed_items) = 0;

  //! Test si la connectivité est à jour
  virtual bool isUpToDate(IItemConnectivity* connectivity) = 0; //! par rapport à la famille source et à la famille target
  virtual bool isUpToDateWithSourceFamily(IItemConnectivity* connectivity) = 0; //! par rapport à la famille source
  virtual bool isUpToDateWithTargetFamily(IItemConnectivity* connectivity) = 0; //! par rapport à la famille target

  //! Enregistre la connectivité comme mise à jour par rapport aux deux familles (source et target)
  virtual void setUpToDate(IItemConnectivity* connectivity) = 0;

  //! Test si la connectivité est à jour
  virtual bool isUpToDate(IIncrementalItemConnectivity* connectivity) = 0; //! par rapport à la famille source et à la famille target
  virtual bool isUpToDateWithSourceFamily(IIncrementalItemConnectivity* connectivity) = 0; //! par rapport à la famille source
  virtual bool isUpToDateWithTargetFamily(IIncrementalItemConnectivity* connectivity) = 0; //! par rapport à la famille target

  //! Enregistre la connectivité comme mise à jour par rapport aux deux familles (source et target)
  virtual void setUpToDate(IIncrementalItemConnectivity* connectivity) = 0;

  //! Mise à jour des items modifiés éventuellement compactés
  virtual void notifyLocalIdChanged(IItemFamily* item_family,
                                    Int32ConstArrayView old_to_new_ids,
                                    Integer nb_item) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ICONNECTIVITYMANAGER_H_ */
