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
  explicit ArrayShape(Span<Int32> v);

 public:

  Int32ConstArrayView values() const { return { m_nb_dim, m_dims.data() }; }
  Int32ConstArrayView dimensions() const { return { m_nb_dim, m_dims.data() }; }
  Int32 nbDimension() const { return m_nb_dim; }
  Int64 totalNbElement() const
  {
    Int64 v = 1;
    for (Int32 i = 0, n = m_nb_dim; i < n; ++i)
      v *= (Int64)m_dims[i];
    return v;
  }
  void setNbDimension(Int32 nb_value);
  void setDimension(Int32 index,Int32 value) { m_dims[index] = value; }

 private:

  Int32 m_nb_dim = 0;
  std::array<Int32, MAX_NB_DIMENSION> m_dims;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
