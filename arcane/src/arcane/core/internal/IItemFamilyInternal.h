// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Internal Arcane part of IItemFamily.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamilyTopologyModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal part of IItemFamily.
 */
class ARCANE_CORE_EXPORT IItemFamilyInternal
{
 public:

  virtual ~IItemFamilyInternal() = default;

 public:

  //! Information on unstructured connectivities
  virtual ItemInternalConnectivityList* unstructuredItemInternalConnectivityList() = 0;

  //! Topology modifier interface.
  virtual IItemFamilyTopologyModifier* topologyModifier() = 0;

  //! Instance of ItemSharedInfo for the family entities
  virtual ItemSharedInfo* commonItemSharedInfo() = 0;

  /*!
   * \brief Indicates the end of allocation.
   *
   * This method should normally only be called by the mesh (IMesh)
   * at the time of allocation.
   *
   * This method is collective.
   */
  virtual void endAllocate() = 0;

  /*!
   * \brief Indicates the end of modification by the mesh.
   *
   * This method should normally only be called by the mesh (IMesh)
   * at the end of an endUpdate().
   *
   * This method is collective.
   */
  virtual void notifyEndUpdateFromMesh() = 0;

  /*!
   * \brief Adds a variable to this family.
   *
   * This method is called by the variable itself and should not
   * be called under other conditions.
   */
  virtual void addVariable(IVariable* var) = 0;

  /*!
   * \brief Removes a variable from this family.
   *
   * This method is called by the variable itself and should not
   * be called under other conditions.
   */
  virtual void removeVariable(IVariable* var) = 0;

  /*!
   * \brief Resizes the variables of this family.
   */
  virtual void resizeVariables(bool force_resize) = 0;

  virtual void addSourceConnectivity(IIncrementalItemSourceConnectivity* connectivity) = 0;
  virtual void addTargetConnectivity(IIncrementalItemTargetConnectivity* connectivity) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
