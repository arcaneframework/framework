// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianGrid.h                                             (C) 2000-2021 */
/*                                                                           */
/* Cartesian grid with nodes, faces, and cells.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANGRID_H
#define ARCANE_CARTESIANMESH_CARTESIANGRID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/v2/CartesianNumbering.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::CartesianMesh::V2
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Encapsulation of a Cartesian grid with cells, nodes, and faces,
 * up to 3 dimensions.
 */
template <typename IdType>
class CartesianGrid
{
 public:
  //! Type for Cartesian triplets (i,j,k) and dimension triplets (ni,nj,nk).
  using IdType3 = IdType[3];

  //! Type of the Cartesian numbering associated with IdType.
  using CartesianNumberingType = CartesianNumbering<IdType>;

  //! Array type of Cartesian numberings across 3 dimensions.
  using CartesianNumberingType3 = CartesianNumberingType[3];

 public:
  //! param[in] ncells_dir Number of cells in each direction.
  //  param[in] dimension  Dimension of the Cartesian grid (up to 3).
  CartesianGrid(const IdType3& ncells_dir, Integer dimension)
  : m_dimension(dimension)
  {

    for (Integer d(0); d < m_dimension; ++d) {
      m_ncells_dir[d] = ncells_dir[d];
      m_nnodes_dir[d] = m_ncells_dir[d] + 1;
    }
    m_cart_num_cell.initNumbering(m_ncells_dir, dimension);

    for (Integer d(m_dimension); d < 3; ++d) {
      m_ncells_dir[d] = 1;
      m_nnodes_dir[d] = 1;
    }
    m_cart_num_node.initNumbering(m_nnodes_dir, dimension);

    // For faces, we will number them according to X, then according to Y and according to Z.
    // Faces are distinguished by their orientations (dnorm).
    IdType nfaces_norm_first = 0; // First face number according to the dnorm normal.
    for (Integer dnorm(0); dnorm < m_dimension; ++dnorm) {

      // Directions orthogonal to the normal.
      Integer d1 = (dnorm + 1) % 3;
      Integer d2 = (dnorm + 2) % 3;

      // In the direction of the normal, we have m_ncells_dir[dnorm]+1 faces
      // and in the directions m_ncells_dir[d] faces
      m_nfaces_norm_dir[dnorm][dnorm] = m_ncells_dir[dnorm] + 1;
      m_nfaces_norm_dir[dnorm][d1] = m_ncells_dir[d1];
      m_nfaces_norm_dir[dnorm][d2] = m_ncells_dir[d2];

      m_cart_num_face[dnorm].initNumbering(m_nfaces_norm_dir[dnorm], dimension, nfaces_norm_first);
      nfaces_norm_first += m_cart_num_face[dnorm].nbItem();
    }

    for (Integer dnorm(m_dimension); dnorm < 3; ++dnorm) {
      m_nfaces_norm_dir[dnorm][0] = 1;
      m_nfaces_norm_dir[dnorm][1] = 1;
      m_nfaces_norm_dir[dnorm][2] = 1;
    }
  }

  //! Read-only reference to the Cartesian numbering for cells.
  const CartesianNumberingType& cartNumCell() const
  {
    return m_cart_num_cell;
  }

  //! Read-only reference to the Cartesian numbering for nodes.
  const CartesianNumberingType& cartNumNode() const
  {
    return m_cart_num_node;
  }

  //! Read-only reference to the Cartesian numbering for faces in direction \a dir.
  const CartesianNumberingType& cartNumFace(Integer dir) const
  {
    ARCANE_ASSERT(dir < m_dimension, ("The direction must be strictly less than the dimension"));
    return m_cart_num_face[dir];
  }

  //! Read-only reference to the 3 Cartesian numberings for faces.
  const CartesianNumberingType3& cartNumFace3() const
  {
    return m_cart_num_face;
  }

  //! Pointer to the Cartesian numbering for cells.
  CartesianNumberingType* cartNumCellPtr()
  {
    return &m_cart_num_cell;
  }

  //! Pointer to the Cartesian numbering for nodes.
  CartesianNumberingType* cartNumNodePtr()
  {
    return &m_cart_num_node;
  }

  //! Pointer to the Cartesian numbering for faces in direction \a dir.
  CartesianNumberingType* cartNumFacePtr(Integer dir)
  {
    ARCANE_ASSERT(dir < m_dimension, ("The direction must be strictly less than the dimension"));
    return &(m_cart_num_face[dir]);
  }

  //! Pointer to the 3 Cartesian numberings for faces.
  CartesianNumberingType3* cartNumFace3Ptr()
  {
    return &m_cart_num_face;
  }

  //! Dimension of the Cartesian mesh.
  Integer dimension() const
  {
    return m_dimension;
  }

 protected:
  IdType3 m_ncells_dir = { 1, 1, 1 }; // Number of cells per direction.
  IdType3 m_nnodes_dir = { 1, 1, 1 }; // Number of nodes per direction.
  IdType3 m_nfaces_norm_dir[3]; //! m_nfaces_norm_dir[dnorm] = dimension of the face grid normal to dnorm.

  Integer m_dimension = 0;

  CartesianNumberingType m_cart_num_cell;
  CartesianNumberingType m_cart_num_node;
  CartesianNumberingType3 m_cart_num_face;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CartesianMesh::V2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
