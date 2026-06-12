// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsOwnerBuilder.h                                         (C) 2000-2025 */
/*                                                                           */
/* Class for calculating entity owners.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITESMOWNERBUILDER_H
#define ARCANE_MESH_ITESMOWNERBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class IMesh;
}
namespace Arcane::mesh
{
class ItemsOwnerBuilderImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic class for calculating entity owners.
 *
 * For all methods, it is assumed that the cell owners
 * are correctly valid and synchronized.
 */
class ARCANE_MESH_EXPORT ItemsOwnerBuilder
{
  class Impl;

 public:

  explicit ItemsOwnerBuilder(IMesh* mesh);
  ~ItemsOwnerBuilder();

 public:

  void computeFacesOwner();
  void computeEdgesOwner();
  void computeNodesOwner();

 private:

  std::unique_ptr<ItemsOwnerBuilderImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
