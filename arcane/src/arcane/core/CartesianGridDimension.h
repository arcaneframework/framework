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
