// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayShape.h                                                (C) 2000-2023 */
/*                                                                           */
/* Represents the shape of an array.                                         */
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
 * All classes in this file are experimental and the API is not
 * finalized. DO NOT USE OUTSIDE OF ARCANE.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Array shape
 */
class ARCANE_UTILS_EXPORT ArrayShape
{
 public:

  static constexpr int MAX_NB_DIMENSION = 8;

  ArrayShape() = default;
  explicit ArrayShape(Span<const Int32> v);

 public:

  //! Rank of the shape
  Int32 nbDimension() const { return m_nb_dim; }

  //! Values of each dimension
  SmallSpan<const Int32> dimensions() const { return { m_dims.data(), m_nb_dim }; }

  //! Number of elements in the index-th dimension
  Int32 dimension(Int32 index) const { return m_dims[index]; }

  //! Total number of elements
  Int64 totalNbElement() const
  {
    Int64 v = 1;
    for (Int32 i = 0, n = m_nb_dim; i < n; ++i)
      v *= (Int64)m_dims[i];
    return v;
  }

  //! Sets the rank of the shape
  void setNbDimension(Int32 nb_value);

  //! Sets the value of the index-th dimension to \a value
  void setDimension(Int32 index, Int32 value) { m_dims[index] = value; }

  //! Sets the number and values of the dimensions
  void setDimensions(Span<const Int32> dims);

  friend std::ostream& operator<<(std::ostream& o, const ArrayShape& s)
  {
    s._print(o);
    return o;
  }
  friend bool operator==(const ArrayShape& s1, const ArrayShape& s2)
  {
    return _isEqual(s1, s2);
  }
  friend bool operator!=(const ArrayShape& s1, const ArrayShape& s2)
  {
    return !_isEqual(s1, s2);
  }

 private:

  Int32 m_nb_dim = 0;
  std::array<Int32, MAX_NB_DIMENSION> m_dims = {};

 private:

  void _set(SmallSpan<const Int32> v);
  static bool _isEqual(const ArrayShape& s1, const ArrayShape& s2);
  void _print(std::ostream& o) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
