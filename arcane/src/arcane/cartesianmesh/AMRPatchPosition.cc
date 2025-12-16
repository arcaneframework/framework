// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPosition.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Position d'un patch AMR d'un maillage cartésien.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPosition.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Math.h"

#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition::
AMRPatchPosition()
: m_level(-2)
, m_overlap_layer_size(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition::
AMRPatchPosition(const AMRPatchPosition& src)
: m_level(src.level())
, m_min_point(src.minPoint())
, m_max_point(src.maxPoint())
, m_overlap_layer_size(src.overlapLayerSize())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition::
~AMRPatchPosition()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AMRPatchPosition::
level() const
{
  return m_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setLevel(Int32 level)
{
  m_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3Type AMRPatchPosition::
minPoint() const
{
  return m_min_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setMinPoint(CartCoord3Type min_point)
{
  m_min_point = min_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3Type AMRPatchPosition::
maxPoint() const
{
  return m_max_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setMaxPoint(CartCoord3Type max_point)
{
  m_max_point = max_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AMRPatchPosition::
overlapLayerSize() const
{
  return m_overlap_layer_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPosition::
setOverlapLayerSize(Int32 layer_size)
{
  m_overlap_layer_size = layer_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3Type AMRPatchPosition::
minPointWithOverlap() const
{
  return m_min_point - m_overlap_layer_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3Type AMRPatchPosition::
maxPointWithOverlap() const
{
  return m_max_point + m_overlap_layer_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 AMRPatchPosition::
nbCells() const
{
  return static_cast<Int64>(m_max_point.x - m_min_point.x) * (m_max_point.y - m_min_point.y) * (m_max_point.z - m_min_point.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<AMRPatchPosition, AMRPatchPosition> AMRPatchPosition::
cut(CartCoordType cut_point, Integer dim) const
{
  CartCoord3Type patch_max_cut = m_max_point;
  CartCoord3Type patch_min_cut = m_min_point;

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

  AMRPatchPosition p0(*this);
  p0.setMaxPoint(patch_max_cut);

  AMRPatchPosition p1(*this);
  p1.setMinPoint(patch_min_cut);

  return { p0, p1 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
canBeFusion(const AMRPatchPosition& other_patch) const
{
  const CartCoord3Type min_point = other_patch.minPoint();
  const CartCoord3Type max_point = other_patch.maxPoint();
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

bool AMRPatchPosition::
fusion(const AMRPatchPosition& other_patch)
{
  if (!canBeFusion(other_patch)) {
    return false;
  }

  const CartCoord3Type min_point = other_patch.minPoint();
  const CartCoord3Type max_point = other_patch.maxPoint();

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

  return true;
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

AMRPatchPosition AMRPatchPosition::
patchUp(Integer dim) const
{
  AMRPatchPosition p;
  p.setLevel(m_level + 1);
  p.setMinPoint(m_min_point * 2);
  if (dim == 2) {
    p.setMaxPoint({ m_max_point.x * 2, m_max_point.y * 2, 1 });
  }
  else {
    p.setMaxPoint(m_max_point * 2);
  }
  p.setOverlapLayerSize(m_overlap_layer_size * 2);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition AMRPatchPosition::
patchDown(Integer dim) const
{
  AMRPatchPosition p;
  p.setLevel(m_level - 1);
  p.setMinPoint(m_min_point / 2);
  if (dim == 2) {
    p.setMaxPoint({ static_cast<CartCoordType>(std::ceil(m_max_point.x / 2.)), static_cast<CartCoordType>(std::ceil(m_max_point.y / 2.)), 1 });
  }
  else {
    p.setMaxPoint({ static_cast<CartCoordType>(std::ceil(m_max_point.x / 2.)), static_cast<CartCoordType>(std::ceil(m_max_point.y / 2.)), static_cast<CartCoordType>(std::ceil(m_max_point.z / 2.)) });
  }
  p.setOverlapLayerSize((m_overlap_layer_size / 2) + 1);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord3Type AMRPatchPosition::
length() const
{
  return m_max_point - m_min_point;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isIn(CartCoordType x, CartCoordType y, CartCoordType z) const
{
  return x >= m_min_point.x && x < m_max_point.x && y >= m_min_point.y && y < m_max_point.y && z >= m_min_point.z && z < m_max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isIn(CartCoord3Type coord) const
{
  return coord.x >= m_min_point.x && coord.x < m_max_point.x && coord.y >= m_min_point.y && coord.y < m_max_point.y && coord.z >= m_min_point.z && coord.z < m_max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isInWithOverlap(CartCoordType x, CartCoordType y, CartCoordType z) const
{
  const CartCoord3Type min_point = minPointWithOverlap();
  const CartCoord3Type max_point = maxPointWithOverlap();
  return x >= min_point.x && x < max_point.x && y >= min_point.y && y < max_point.y && z >= min_point.z && z < max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isInWithOverlap(CartCoord3Type coord) const
{
  const CartCoord3Type min_point = minPointWithOverlap();
  const CartCoord3Type max_point = maxPointWithOverlap();
  return coord.x >= min_point.x && coord.x < max_point.x && coord.y >= min_point.y && coord.y < max_point.y && coord.z >= min_point.z && coord.z < max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isInWithOverlap(CartCoordType x, CartCoordType y, CartCoordType z, Integer overlap) const
{
  const CartCoord3Type min_point = m_min_point - overlap;
  const CartCoord3Type max_point = m_max_point + overlap;
  return x >= min_point.x && x < max_point.x && y >= min_point.y && y < max_point.y && z >= min_point.z && z < max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
isInWithOverlap(CartCoord3Type coord, Integer overlap) const
{
  const CartCoord3Type min_point = m_min_point - overlap;
  const CartCoord3Type max_point = m_max_point + overlap;
  return coord.x >= min_point.x && coord.x < max_point.x && coord.y >= min_point.y && coord.y < max_point.y && coord.z >= min_point.z && coord.z < max_point.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPosition::
haveIntersection(const AMRPatchPosition& other) const
{
  return (
  (other.level() == level()) &&
  (other.maxPoint().x > minPoint().x && maxPoint().x > other.minPoint().x) &&
  (other.maxPoint().y > minPoint().y && maxPoint().y > other.minPoint().y) &&
  (other.maxPoint().z > minPoint().z && maxPoint().z > other.minPoint().z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
