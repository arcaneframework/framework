// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayShape.h                                                (C) 2000-2022 */
/*                                                                           */
/* Représente la forme d'un tableau.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYSHAPE_H
#define ARCANE_UTILS_ARRAYSHAPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ArrayView.h"

#include <array>

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Forme d'un tableau
 */
class ARCANE_UTILS_EXPORT ArrayShape
{
 public:

  static constexpr int MAX_NB_DIMENSION = 8;

  ArrayShape() = default;
  explicit ArrayShape(Span<const Int32> v);

 public:

  //! Rang de la forme
  Int32 nbDimension() const { return m_nb_dim; }

  //! Valeurs de chaque dimension
  SmallSpan<const Int32> dimensions() const { return { m_dims.data(), m_nb_dim }; }

  //! Nombre d'élements de la index-ème dimension
  Int32 dimension(Int32 index) const { return m_dims[index]; }

  //! Nombre total d'élements
  Int64 totalNbElement() const
  {
    Int64 v = 1;
    for (Int32 i = 0, n = m_nb_dim; i < n; ++i)
      v *= (Int64)m_dims[i];
    return v;
  }

  //! Positionne le rang de la forme
  void setNbDimension(Int32 nb_value);

  //! Positionne la valeur de la index-ème dimension à \a value
  void setDimension(Int32 index, Int32 value) { m_dims[index] = value; }

  //! Positionne le nombre et la valeur des dimensions
  void setDimensions(Span<const Int32> dims);

 private:

  Int32 m_nb_dim = 0;
  std::array<Int32, MAX_NB_DIMENSION> m_dims = { };

 private:

  void _set(SmallSpan<const Int32> v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
