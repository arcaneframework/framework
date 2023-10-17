// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshDimension.h                                    (C) 2000-2023 */
/*                                                                           */
/* Informations sur les dimensions d'une grille cartésienne.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CARTESIANGRIDDIMENSION_H
#define ARCANE_CORE_CARTESIANGRIDDIMENSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Vector3.h"

#include "arcane/core/ArcaneTypes.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// Temporaire pour les classes friend
namespace mesh
{
  class DynamicMeshCartesian2DBuilder;
  class DynamicMeshCartesian3DBuilder;
  class CartesianFaceUniqueIdBuilder;
} // namespace mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur les dimensions d'une grille cartésienne.
 *
 * Cette classe permet d'obtenir à partir du nombre de mailles dans
 * chaque direction les différentes informations sur les dimensions
 * du maillage.
 */
class ARCANE_CORE_EXPORT CartesianGridDimension
{
  friend mesh::DynamicMeshCartesian2DBuilder;
  friend mesh::DynamicMeshCartesian3DBuilder;
  friend mesh::CartesianFaceUniqueIdBuilder;
  friend class CartesianMeshUniqueIdRenumbering;
  friend class CartesianMeshCoarsening;
  friend class CartesianMeshCoarsening2;

 private:

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'un noeud en fonction
   * de sa position dans la grille.
   */
  class NodeUniqueIdComputer2D
  {
   public:

    NodeUniqueIdComputer2D(Int64 base_offset, Int64 nb_node_x)
    : m_base_offset(base_offset)
    , m_nb_node_x(nb_node_x)
    {}

   public:

    Int64 compute(Int64 x, Int64 y)
    {
      return m_base_offset + x + y * m_nb_node_x;
    }

    std::array<Int64, 4> computeForCell(Int64 x, Int64 y)
    {
      std::array<Int64, 4> node_uids;
      node_uids[0] = compute(x + 0, y + 0);
      node_uids[1] = compute(x + 1, y + 0);
      node_uids[2] = compute(x + 1, y + 1);
      node_uids[3] = compute(x + 0, y + 1);
      return node_uids;
    }

   private:

    Int64 m_base_offset;
    Int64 m_nb_node_x;
  };

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'une maille en fonction
   * de sa position dans la grille.
   */
  class CellUniqueIdComputer2D
  {
   public:

    CellUniqueIdComputer2D(Int64 base_offset, Int64 all_nb_cell_x)
    : m_base_offset(base_offset)
    , m_all_nb_cell_x(all_nb_cell_x)
    {}

   public:

    Int64 compute(Int64 x, Int64 y)
    {
      return m_base_offset + x + y * m_all_nb_cell_x;
    }
    Int64x3 compute(Int64 unique_id)
    {
      Int64 uid = unique_id - m_base_offset;
      const Int64 y = uid / m_all_nb_cell_x;
      const Int64 x = uid % m_all_nb_cell_x;
      return Int64x3(x, y, 0);
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_cell_x;
  };

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'une face en fonction
   * de sa position dans la grille.
   */
  class FaceUniqueIdComputer2D
  {
   public:

    FaceUniqueIdComputer2D(Int64 base_offset, Int64 nb_cell_x, Int64 nb_face_x, Int64 nb_face_dir_x)
    : m_base_offset(base_offset)
    , m_nb_cell_x(nb_cell_x)
    , m_nb_face_x(nb_face_x)
    , m_nb_face_dir_x(nb_face_dir_x)
    {}

   public:

    //! Calcule les uniqueIds() des 4 faces de la mailles de coordonnées topologique (x,y)
    std::array<Int64, 4> computeForCell(Int64 x, Int64 y)
    {
      std::array<Int64, 4> face_uids;

      // Faces selon Y
      face_uids[0] = m_base_offset + (x + 0) + ((y + 0) * m_nb_cell_x) + m_nb_face_dir_x;
      face_uids[2] = m_base_offset + (x + 0) + ((y + 1) * m_nb_cell_x) + m_nb_face_dir_x;

      // Faces selon X
      face_uids[1] = m_base_offset + (x + 1) + (y + 0) * m_nb_face_x;
      face_uids[3] = m_base_offset + (x + 0) + (y + 0) * m_nb_face_x;

      return face_uids;
    }

   private:

    Int64 m_base_offset;
    Int64 m_nb_cell_x;
    Int64 m_nb_face_x;
    Int64 m_nb_face_dir_x;
  };

  /*!
   * \brief Classe pour calculer en 3D le uniqueId() d'un noeud en fonction
   * de sa position dans la grille.
   */
  class NodeUniqueIdComputer3D
  {
   public:

    NodeUniqueIdComputer3D(Int64 base_offset, Int64 nb_node_x, Int64 nb_node_xy)
    : m_base_offset(base_offset)
    , m_nb_node_x(nb_node_x)
    , m_nb_node_xy(nb_node_xy)
    {}

   public:

    Int64 compute(Int64 x, Int64 y, Int64 z)
    {
      return m_base_offset + x + y * m_nb_node_x + z * m_nb_node_xy;
    }

    std::array<Int64, 8> computeForCell(Int64 x, Int64 y, Int64 z)
    {
      std::array<Int64, 8> node_uids;
      node_uids[0] = compute(x + 0, y + 0, z + 0);
      node_uids[1] = compute(x + 1, y + 0, z + 0);
      node_uids[2] = compute(x + 1, y + 1, z + 0);
      node_uids[3] = compute(x + 0, y + 1, z + 0);
      node_uids[4] = compute(x + 0, y + 0, z + 1);
      node_uids[5] = compute(x + 1, y + 0, z + 1);
      node_uids[6] = compute(x + 1, y + 1, z + 1);
      node_uids[7] = compute(x + 0, y + 1, z + 1);
      return node_uids;
    }

   private:

    Int64 m_base_offset;
    Int64 m_nb_node_x;
    Int64 m_nb_node_xy;
  };

  /*!
   * \brief Classe pour calculer en 3D le uniqueId() d'une maille en fonction
   * de sa position dans la grille.
   */
  class CellUniqueIdComputer3D
  {
   public:

    CellUniqueIdComputer3D(Int64 base_offset, Int64 all_nb_cell_x, Int64 all_nb_cell_xy)
    : m_base_offset(base_offset)
    , m_all_nb_cell_x(all_nb_cell_x)
    , m_all_nb_cell_xy(all_nb_cell_xy)
    {}

   public:

    //! Calcul le uniqueId() en fonction des coordonnées
    Int64 compute(Int64 x, Int64 y, Int64 z)
    {
      return m_base_offset + x + y * m_all_nb_cell_x + z * m_all_nb_cell_xy;
    }
    //! Calcul les coordonnées en fonction du uniqueId().
    Int64x3 compute(Int64 unique_id)
    {
      Int64 uid = unique_id - m_base_offset;
      Int64 z = uid / m_all_nb_cell_xy;
      Int64 v = uid - (z * m_all_nb_cell_xy);
      Int64 y = v / m_all_nb_cell_x;
      Int64 x = v % m_all_nb_cell_x;
      return Int64x3(x, y, z);
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_cell_x;
    Int64 m_all_nb_cell_xy;
  };

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'une face en fonction
   * de sa position dans la grille.
   */
  class FaceUniqueIdComputer3D
  {
   public:

    FaceUniqueIdComputer3D(Int64 base_offset, Int64 nb_cell_x, Int64 nb_face_x, Int64x3 nb_face_dir,
                           Int64 total_nb_face_xy, Int64 total_nb_face_x)
    : m_base_offset(base_offset)
    , m_nb_cell_x(nb_cell_x)
    , m_nb_face_x(nb_face_x)
    , m_nb_face_dir(nb_face_dir)
    , m_total_nb_face_xy(total_nb_face_xy)
    , m_total_nb_face_x(total_nb_face_x)
    {}

   public:

    //! Calcule les uniqueIds() des 6 faces de la mailles de coordonnées topologique (x,y,z)
    std::array<Int64, 6> computeForCell(Int64 x, Int64 y, Int64 z)
    {
      std::array<Int64, 6> face_uids;

      // Faces selon Z
      face_uids[0] = (x + 0) + ((y + 0) * m_nb_cell_x) + ((z + 0) * m_nb_face_dir.z) + m_total_nb_face_xy;
      face_uids[3] = (x + 0) + ((y + 0) * m_nb_cell_x) + ((z + 1) * m_nb_face_dir.z) + m_total_nb_face_xy;

      // Faces selon X
      face_uids[1] = (x + 0) + ((y + 0) * m_nb_face_x) + ((z + 0) * m_nb_face_dir.x);
      face_uids[4] = (x + 1) + ((y + 0) * m_nb_face_x) + ((z + 0) * m_nb_face_dir.x);

      // Faces selon Y
      face_uids[2] = (x + 0) + ((y + 0) * m_nb_cell_x) + ((z + 0) * m_nb_face_dir.y) + m_total_nb_face_x;
      face_uids[5] = (x + 0) + ((y + 1) * m_nb_cell_x) + ((z + 0) * m_nb_face_dir.y) + m_total_nb_face_x;

      return face_uids;
    }

   private:

    Int64 m_base_offset;
    Int64 m_nb_cell_x;
    Int64 m_nb_face_x;
    Int64x3 m_nb_face_dir;
    Int64 m_total_nb_face_xy;
    Int64 m_total_nb_face_x;
  };

 public:

  CartesianGridDimension() = default;
  CartesianGridDimension(Int64 nb_cell_x, Int64 nb_cell_y);
  CartesianGridDimension(Int64 nb_cell_x, Int64 nb_cell_y, Int64 nb_cell_z);
  explicit CartesianGridDimension(std::array<Int64, 2> dims);
  explicit CartesianGridDimension(std::array<Int64, 3> dims);
  explicit CartesianGridDimension(std::array<Int32, 2> dims);
  explicit CartesianGridDimension(std::array<Int32, 3> dims);

 public:

  //! Nombre de mailles dans chaque direction
  constexpr Int64x3 nbCell() const { return m_nb_cell; }

  //! Nombre de noeuds dans chaque direction
  constexpr Int64x3 nbNode() const { return m_nb_node; }

  //! Nombre de faces dans chaque direction
  constexpr Int64x3 nbFace() const { return m_nb_face; }

  //! Nombre total de faces parallèles à une direction donnée
  constexpr Int64x3 nbFaceParallelToDirection() const { return m_nb_face_oriented; }

  //! Nombre total de mailles
  constexpr Int64 totalNbCell() const { return m_total_nb_cell; }

 private:

  //! Instance pour calculer les uniqueId() des noeuds pour cette grille
  NodeUniqueIdComputer2D getNodeComputer2D(Int64 offset) const
  {
    return { offset, m_nb_node.x };
  }
  //! Instance pour calculer les uniqueId() des noeuds pour cette grille
  NodeUniqueIdComputer3D getNodeComputer3D(Int64 offset) const
  {
    return { offset, m_nb_node.x, m_nb_node.x * m_nb_node.y };
  }
  //! Instance pour calculer les uniqueId() des mailles pour cette grille
  CellUniqueIdComputer2D getCellComputer2D(Int64 offset) const
  {
    return { offset, m_nb_cell.x };
  }
  //! Instance pour calculer les uniqueId() des mailles pour cette grille
  CellUniqueIdComputer3D getCellComputer3D(Int64 offset) const
  {
    return { offset, m_nb_cell.x, m_nb_cell.x * m_nb_cell.y };
  }
  //! Instance pour calculer les uniqueId() des faces pour cette grille
  FaceUniqueIdComputer2D getFaceComputer2D(Int64 offset) const
  {
    Int64x3 nb_face_dir = nbFaceParallelToDirection();
    return { offset, m_nb_cell.x, m_nb_face.x, nb_face_dir.x };
  }
  //! Instance pour calculer les uniqueId() des faces pour cette grille
  FaceUniqueIdComputer3D getFaceComputer3D(Int64 offset) const
  {
    Int64x3 nb_face_dir = nbFaceParallelToDirection();
    Int64 total_nb_face_xy = (nb_face_dir.x + nb_face_dir.y) * m_nb_cell.z;
    Int64 total_nb_face_x = (nb_face_dir.x * m_nb_cell.z);
    return { offset, m_nb_cell.x, m_nb_face.x, nb_face_dir, total_nb_face_xy, total_nb_face_x };
  }

 private:

  //! Nombre de mailles dans chaque direction
  Int64x3 m_nb_cell;
  //! Nombre de noeuds dans chaque direction
  Int64x3 m_nb_node;
  //! Nombre de faces dans chaque direction
  Int64x3 m_nb_face;
  //! Nombre total de faces dans une orientation donnée
  Int64x3 m_nb_face_oriented;

  Int64 m_nb_cell_xy = 0;
  Int64 m_total_nb_cell = 0;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
