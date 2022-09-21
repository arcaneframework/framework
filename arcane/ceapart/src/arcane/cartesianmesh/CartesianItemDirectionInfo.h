// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Infos sur les entités devant et derrière une entité.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANITEMDIRECTIONINFO_H
#define ARCANE_CARTESIANMESH_CARTESIANITEMDIRECTIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
/*!
 * \brief Structure interne contenant l'entité devant et derriére dans une
 * direction.
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

  CartesianItemDirectionInfo(Int32 next_id, Int32 prev_id)
  : m_next_lid(next_id)
  , m_previous_lid(prev_id)
  {}

 private:

  //! entité après l'entité courante dans la direction
  Int32 m_next_lid = NULL_ITEM_LOCAL_ID;
  //! entité avant l'entité courante dans la direction
  Int32 m_previous_lid = NULL_ITEM_LOCAL_ID;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
