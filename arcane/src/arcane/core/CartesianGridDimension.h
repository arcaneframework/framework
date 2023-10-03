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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// Temporaire pour les classes friend
namespace mesh
{
  class DynamicMeshCartesian2DBuilder;
  class DynamicMeshCartesian3DBuilder;
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

 private:

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'un noeud en fonction
   * de sa position dans la grille.
   */
  class NodeUniqueIdComputer2D
  {
   public:

    NodeUniqueIdComputer2D(Int64 base_offset, Int64 all_nb_node_x)
    : m_base_offset(base_offset)
    , m_all_nb_node_x(all_nb_node_x)
    {}

   public:

    Int64 compute(Int32 x, Int32 y)
    {
      return m_base_offset + x + y * m_all_nb_node_x;
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_node_x;
  };

  /*!
   * \brief Classe pour calculer en 2D le uniqueId() d'un noeud en fonction
   * de sa position dans la grille.
   */
  class NodeUniqueIdComputer3D
  {
   public:

    NodeUniqueIdComputer3D(Int64 base_offset, Int64 all_nb_node_x, Int64 all_nb_node_xy)
    : m_base_offset(base_offset)
    , m_all_nb_node_x(all_nb_node_x)
    , m_all_nb_node_xy(all_nb_node_xy)
    {}

   public:

    Int64 compute(Int32 x, Int32 y, Int32 z)
    {
      return m_base_offset + x + y * m_all_nb_node_x + z * m_all_nb_node_xy;
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_node_x;
    Int64 m_all_nb_node_xy;
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
