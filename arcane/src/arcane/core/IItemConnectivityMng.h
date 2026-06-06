// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityMng.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface for the entity connectivity manager.                            */
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

  /** Class destructor */
  virtual ~IItemConnectivityMng() = default;

 public:

  //! Registering a connectivity
  virtual void registerConnectivity(IItemConnectivity* connectivity) = 0;
  virtual void unregisterConnectivity(IItemConnectivity* connectivity) = 0;

  virtual void registerConnectivity(IIncrementalItemConnectivity* connectivity) = 0;
  virtual void unregisterConnectivity(IIncrementalItemConnectivity* connectivity) = 0;

  /*!
   * \brief Creation of a synchronization object for a connectivity.
   *
   * If the method has already been called for this connectivity,
   * a new synchronizer is created and the previous one is destroyed.
   */
  virtual IItemConnectivitySynchronizer* createSynchronizer(IItemConnectivity* connectivity,
                                                            IItemConnectivityGhostPolicy* connectivity_ghost_policy) = 0;
  virtual IItemConnectivitySynchronizer* getSynchronizer(IItemConnectivity* connectivity) = 0;

  //! Registering modifications of an item family
  virtual void setModifiedItems(IItemFamily* family,
                                Int32ConstArrayView added_items,
                                Int32ConstArrayView removed_items) = 0;

  //! Retrieval of modified items to update a connectivity
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

  //! Test if the connectivity is up to date
  virtual bool isUpToDate(IItemConnectivity* connectivity) = 0; //! relative to the source family and the target family
  virtual bool isUpToDateWithSourceFamily(IItemConnectivity* connectivity) = 0; //! relative to the source family
  virtual bool isUpToDateWithTargetFamily(IItemConnectivity* connectivity) = 0; //! relative to the target family

  //! Registers the connectivity as up to date relative to both families (source and target)
  virtual void setUpToDate(IItemConnectivity* connectivity) = 0;

  //! Test if the connectivity is up to date
  virtual bool isUpToDate(IIncrementalItemConnectivity* connectivity) = 0; //! relative to the source family and the target family
  virtual bool isUpToDateWithSourceFamily(IIncrementalItemConnectivity* connectivity) = 0; //! relative to the source family
  virtual bool isUpToDateWithTargetFamily(IIncrementalItemConnectivity* connectivity) = 0; //! relative to the target family

  //! Registers the connectivity as up to date relative to both families (source and target)
  virtual void setUpToDate(IIncrementalItemConnectivity* connectivity) = 0;

  //! Update of modified items, possibly compacted
  virtual void notifyLocalIdChanged(IItemFamily* item_family,
                                    Int32ConstArrayView old_to_new_ids,
                                    Integer nb_item) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
