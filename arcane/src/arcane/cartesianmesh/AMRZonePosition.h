// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRZonePosition.h                                           (C) 2000-2026 */
/*                                                                           */
/* Definition of a 2D or 3D zone of a mesh.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRBOXPOSITION_H
#define ARCANE_CARTESIANMESH_AMRBOXPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/Real3.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Class allowing the definition of a mesh zone.
 */
class ARCANE_CARTESIANMESH_EXPORT AMRZonePosition
{
 public:

  /*!
   * \brief 3D zone constructor.
   * \param position The geometric position of the zone.
   * \param length The size of the zone.
   */
  AMRZonePosition(const Real3& position, const Real3& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(true)
  {}

  /*!
   * \brief 2D zone constructor.
   * \param position The geometric position of the zone.
   * \param length The size of the zone.
   */
  AMRZonePosition(const Real2& position, const Real2& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(false)
  {}

 public:

  /*!
   * \brief Method allowing retrieval of the zone's position.
   * \return The geometric position of the zone.
   */
  Real3 position() const
  {
    return m_position;
  }

  /*!
   * \brief Method allowing retrieval of the zone's size.
   * \return The size of the zone.
   */
  Real3 length() const
  {
    return m_length;
  }

  /*!
   * \brief Method allowing determination if it is a 2D or 3D zone.
   * \return True if it is a 3D zone.
   */
  bool is3d() const
  {
    return m_is_3d;
  }

  /*!
   * \brief Method allowing retrieval of the meshes included in the zone.
   * \param mesh The mesh.
   * \param cells_local_id The array that will contain the localIds of the zone's meshes.
   *                       Note: the array will first be cleared.
   */
  void cellsInPatch(IMesh* mesh, UniqueArray<Int32>& cells_local_id) const;

  /*!
   * \brief Method allowing retrieval of the meshes included in the zone.
   * An AMRPatchPosition object designating the patch position is also filled.
   * \param mesh The mesh.
   * \param cells_local_id The array that will contain the localIds of the zone's meshes.
   *                       Note: the array will first be cleared.
   * \param position [OUT] The patch position.
   */
  void cellsInPatch(ICartesianMesh* mesh, UniqueArray<Int32>& cells_local_id, AMRPatchPosition& position) const;

  /*!
   * \brief Method allowing conversion of this AMRZonePosition into
   * AMRPatchPosition.
   *
   * \warning The size of the overlap mesh layer is not
   * correctly defined! This must be done after calling this method.
   *
   * \param mesh The Cartesian mesh.
   * \return The corresponding AMRPatchPosition.
   */
  AMRPatchPosition toAMRPatchPosition(ICartesianMesh* mesh) const;

 private:

  Real3 m_position;
  Real3 m_length;
  bool m_is_3d;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
