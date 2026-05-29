// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Information about the entities in front of and behind an entity.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANITEMDIRECTIONINFO_H
#define ARCANE_CARTESIANMESH_CARTESIANITEMDIRECTIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal structure containing the entity in front and behind in a direction.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianItemDirectionInfo
{
 public:

  friend class Arcane::FaceDirectionMng;
  friend class Arcane::CellDirectionMng;
  friend class Arcane::CartesianMeshImpl;
  friend class Arcane::CartesianMeshPatch;

 public:

  CartesianItemDirectionInfo() = default;

 private:

  CartesianItemDirectionInfo(ItemLocalId next_id, ItemLocalId prev_id)
  : m_next_lid(next_id)
  , m_previous_lid(prev_id)
  {}

 private:

  //! entity after the current entity in the direction
  ItemLocalId m_next_lid;
  //! entity before the current entity in the direction
  ItemLocalId m_previous_lid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
