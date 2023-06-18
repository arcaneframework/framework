// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshDimension.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Informations sur les dimensions d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CartesianGridDimension.h"

#include "arcane/utils/Math.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(Int64 nb_cell_x, Int64 nb_cell_y, Int64 nb_cell_z)
: m_nb_cell{ math::max(nb_cell_x, 0), math::max(nb_cell_y, 0), math::max(nb_cell_z, 0) }
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(Int64 nb_cell_x, Int64 nb_cell_y)
: CartesianGridDimension(nb_cell_x, nb_cell_y, 0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(std::array<Int64, 2> dims)
: CartesianGridDimension(dims[0], dims[1])
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(std::array<Int64, 3> dims)
: CartesianGridDimension(dims[0], dims[1], dims[2])
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(std::array<Int32, 2> dims)
: CartesianGridDimension(dims[0], dims[1])
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianGridDimension::
CartesianGridDimension(std::array<Int32, 3> dims)
: CartesianGridDimension(dims[0], dims[1], dims[2])
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianGridDimension::
_init()
{
  const bool is_dim3 = (m_nb_cell.z > 0);
  const bool is_dim2_or_3 = is_dim3 || (m_nb_cell.y > 0);

  m_nb_face.x = m_nb_cell.x + 1;
  m_nb_node.x = m_nb_cell.x + 1;
  if (is_dim2_or_3) {
    m_nb_face.y = m_nb_cell.y + 1;
    m_nb_node.y = m_nb_cell.y + 1;
  }
  if (is_dim3) {
    m_nb_face.z = m_nb_cell.z + 1;
    m_nb_node.z = m_nb_cell.z + 1;
  }

  m_nb_face_oriented.x = m_nb_face.x * m_nb_cell.y;
  m_nb_face_oriented.y = m_nb_face.y * m_nb_cell.x;
  m_nb_face_oriented.z = m_nb_cell.x * m_nb_cell.y;

  m_nb_cell_xy = m_nb_cell.x * m_nb_cell.y;
  m_total_nb_cell = m_nb_cell_xy * m_nb_cell.z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
