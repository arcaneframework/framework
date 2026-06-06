// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshChecker.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface for mesh verification methods.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCHECKER_H
#define ARCANE_CORE_IMESHCHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for mesh verification methods.
 */
class IMeshChecker
{
 public:

  virtual ~IMeshChecker() = default; //!< Frees resources

 public:

  //! Associated mesh
  virtual IMesh* mesh() = 0;

  /*!
   * \brief Sets the mesh verification level.
   *
   * 0 - tests disabled
   * 1 - partial tests, after endUpdate()
   * 2 - full tests, after endUpdate()
   */
  virtual void setCheckLevel(Integer level) = 0;

  //! Current verification level
  virtual Integer checkLevel() const = 0;

  /*!
   * \brief Verification of the validity of internal mesh structures (internal).
   */
  virtual void checkValidMesh() = 0;

  /*!
   * \brief Verification of mesh validity.
   *
   * This is a global verification across all subdomains.
   *
   * It checks, in particular, that the connectivity is consistent between
   * subdomains.
   *
   * The verification can be quite CPU time-intensive.
   * This method is collective.
   */
  virtual void checkValidMeshFull() = 0;

  /*!
   * \brief Checks that subdomains are correctly replicated.
   *
   * The following checks are performed:
   * - same entity families and same values for these families.
   * - same mesh node coordinates.
   */
  virtual void checkValidReplication() = 0;

  /*!
   * \brief Checks variable synchronization.
   *
   * Checks for each variable that its values on ghost entities are
   * the same as the value on the entity's owning subdomain.
   *
   * Variables on particles are not compared.
   *
   * Raises a FatalErrorException in case of error.
   */
  virtual void checkVariablesSynchronization() = 0;

  /*!
   * \brief Checks synchronization on entity groups.
   *
   * Checks for each group of each family (other than particles)
   * that the entities are the same on each subdomain.
   *
   * Raises a FatalErrorException in case of error.
   */
  virtual void checkItemGroupsSynchronization() = 0;

  /*!
   * \brief Indicates whether entity owner verification is active.
   *
   * This verification is performed when calling checkValidConnectivity().
   * If it is active, we check that the nodes, edges, and
   * faces have the same owner as one of the meshes they are
   * connected to.
   *
   * This is always the case if the owners are managed by %Arcane
   * and it is therefore preferable to always perform this verification to
   * ensure consistency of information in parallel. However, if the
   * owner management is done by the user, it is possible
   * to disable this verification.
   */
  virtual void setIsCheckItemsOwner(bool v) = 0;

  //! Indicates whether entity owner verification is active (true by default)
  virtual bool isCheckItemsOwner() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
