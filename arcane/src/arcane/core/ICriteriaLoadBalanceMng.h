// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICriteriaLoadBalanceMng.h                                   (C) 2000-2024 */
/*                                                                           */
/* Interface for a load balance criteria manager for meshes.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICRITERIALOADBALANCEMNG_H
#define ARCANE_CORE_ICRITERIALOADBALANCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class interface allowing criteria to be added to adjust
 *        the load balance.
 */
class ICriteriaLoadBalanceMng
{
 public:

  virtual ~ICriteriaLoadBalanceMng() = default; //!< Frees resources.

 public:

  /*!
   * \brief Method allowing a criterion to be added for each cell.
   *
   * \param count A cell variable with a weight per cell.
   */
  virtual void addCriterion(VariableCellInt32& count) =0;

  /*!
   * \brief Method allowing a criterion to be added for each cell.
   *
   * \param count A cell variable with a weight per cell.
   */
  virtual void addCriterion(VariableCellReal& count) =0;

  // TODO Understand how PartitionerMemoryInfo works
  /*!
   * \brief Method allowing a criterion to be added for each cell.
   *
   * \param count A cell variable with a weight per cell.
   * \param entity The entity type linked to this criterion.
   */
  virtual void addMass(VariableCellInt32& count, const String& entity) = 0;

  /*!
   * \brief Method allowing a criterion to be added for each face.
   *
   * \param count A face variable with a weight per face.
   * \param entity The entity type linked to this criterion.
   */
  virtual void addCommCost(VariableFaceInt32& count, const String& entity) = 0;

  /*!
   * \brief Method allowing the criteria already added to be cleared.
   */
  virtual void reset() =0;

  /*!
   * \brief Method allowing to specify if the data mass of each
   *        cell is a criterion for load balance.
   *
   * \param active true if the data mass must be a criterion.
   */
  virtual void setMassAsCriterion(bool active) = 0;

  /*!
   * \brief Method allowing to specify if the number of cells in a
   *        subdomain must be a criterion for load balance.
   *
   * \param active true if the number of cells must be a criterion.
   */
  virtual void setNbCellsAsCriterion(bool active) = 0;

  /*!
   * \brief Method allowing to specify if the mass of communications
   *        between cells must be a criterion for load balance.
   *
   * \param active true if the mass of communications must be a criterion.
   */
  virtual void setCellCommContrib(bool active) = 0;

  /*!
   * \brief
   * \param active
   */
  virtual void setComputeComm(bool active) = 0;

  /*!
   * \brief Method allowing to retrieve the number of criteria already registered.
   *
   * \return The number of criteria.
   */
  virtual Integer nbCriteria() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
