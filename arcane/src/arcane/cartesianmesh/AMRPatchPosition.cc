// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPosition.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPosition.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition::
AMRPatchPosition()
: m_level(-2)
{

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition::
~AMRPatchPosition()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer AMRPatchPosition::
level() const
{
  return m_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setLevel(Integer level)
{
  m_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64x3 AMRPatchPosition::
minPoint() const
{
  return m_min_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setMinPoint(Int64x3 min_point)
{
  m_min_point = min_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64x3 AMRPatchPosition::
maxPoint() const
{
  return m_max_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setMaxPoint(Int64x3 max_point)
{
  m_max_point = max_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isIn(Int64 x, Int64 y, Int64 z) const
{
  return x >= m_min_point.x && x < m_max_point.x && y >= m_min_point.y && y < m_max_point.y && z >= m_min_point.z && z < m_max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 AMRPatchPosition::
nbCells() const
{
  return (m_max_point.x - m_min_point.x) * (m_max_point.y - m_min_point.y) * (m_max_point.z - m_min_point.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<AMRPatchPosition, AMRPatchPosition> AMRPatchPosition::
cut(Int64 cut_point, Integer dim) const
{
  Int64x3 patch_max_cut = m_max_point;
  Int64x3 patch_min_cut = m_min_point;

  if (dim == MD_DirX) {
    patch_max_cut.x = cut_point;
    patch_min_cut.x = cut_point;
  }
  else if (dim == MD_DirY) {
    patch_max_cut.y = cut_point;
    patch_min_cut.y = cut_point;
  }
  else {
    patch_max_cut.z = cut_point;
    patch_min_cut.z = cut_point;
  }

  AMRPatchPosition p0;
  p0.setLevel(m_level);
  p0.setMinPoint(m_min_point);
  p0.setMaxPoint(patch_max_cut);

  AMRPatchPosition p1;
  p1.setLevel(m_level);
  p1.setMinPoint(patch_min_cut);
  p1.setMaxPoint(m_max_point);

  return { p0, p1 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
canBeFusion(const AMRPatchPosition& other_patch) const
{
  const Int64x3 min_point = other_patch.minPoint();
  const Int64x3 max_point = other_patch.maxPoint();
  return m_level == other_patch.level() &&
  (((m_min_point.x == max_point.x || m_max_point.x == min_point.x) &&
    (m_min_point.y == min_point.y && m_max_point.y == max_point.y) &&
    (m_min_point.z == min_point.z && m_max_point.z == max_point.z)) ||

   ((m_min_point.x == min_point.x && m_max_point.x == max_point.x) &&
    (m_min_point.y == max_point.y || m_max_point.y == min_point.y) &&
    (m_min_point.z == min_point.z && m_max_point.z == max_point.z)) ||

   ((m_min_point.x == min_point.x && m_max_point.x == max_point.x) &&
    (m_min_point.y == min_point.y && m_max_point.y == max_point.y) &&
    (m_min_point.z == max_point.z || m_max_point.z == min_point.z)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
fusion(const AMRPatchPosition& other_patch)
{
  const Int64x3 min_point = other_patch.minPoint();
  const Int64x3 max_point = other_patch.maxPoint();

  if (m_min_point.x > min_point.x) {
    m_min_point.x = min_point.x;
  }
  else if (m_max_point.x < max_point.x) {
    m_max_point.x = max_point.x;
  }

  else if (m_min_point.y > min_point.y) {
    m_min_point.y = min_point.y;
  }
  else if (m_max_point.y < max_point.y) {
    m_max_point.y = max_point.y;
  }

  else if (m_min_point.z > min_point.z) {
    m_min_point.z = min_point.z;
  }
  else if (m_max_point.z < max_point.z) {
    m_max_point.z = max_point.z;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isNull() const
{
  return m_level == -2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
