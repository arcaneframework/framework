// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatVarIndex.h                                               (C) 2000-2024 */
/*                                                                           */
/* Index on material variables.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATVARINDEX_H
#define ARCANE_CORE_MATERIALS_MATVARINDEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Represents an index on material and environment variables.
 *
 * The index includes 2 values:
 * - the first (arrayIndex()) is the number in the list of the variable's arrays.
 * - the second (valueIndex()) is the index in the array of values of this variable.
 *
 * \note For performance reasons, the default constructor
 * does not initialize the members of this class. Therefore, reset() must be called
 * to initialize to an invalid value.
 */
class MatVarIndex
{
 public:

  constexpr ARCCORE_HOST_DEVICE MatVarIndex(Int32 array_index, Int32 value_index)
  : m_array_index(array_index)
  , m_value_index(value_index)
  {
  }
  ARCCORE_HOST_DEVICE MatVarIndex() {}

 public:

  //! Returns the index of the value array in the list of variables.
  constexpr ARCCORE_HOST_DEVICE Int32 arrayIndex() const { return m_array_index; }

  //! Returns the index in the value array
  constexpr ARCCORE_HOST_DEVICE Int32 valueIndex() const { return m_value_index; }

  //! Sets the index
  constexpr ARCCORE_HOST_DEVICE void setIndex(Int32 array_index, Int32 value_index)
  {
    m_array_index = array_index;
    m_value_index = value_index;
  }

  //! Sets the entity to the null instance.
  constexpr ARCCORE_HOST_DEVICE void reset()
  {
    m_array_index = (-1);
    m_value_index = (-1);
  }

  //! Indicates if the instance represents the null entity
  constexpr ARCCORE_HOST_DEVICE bool null() const
  {
    return m_value_index == (-1);
  }

  //! Indicates if the instance represents the null entity
  constexpr ARCCORE_HOST_DEVICE bool isNull() const
  {
    return m_value_index == (-1);
  }

  //! Comparison operator
  constexpr ARCCORE_HOST_DEVICE friend bool
  operator==(MatVarIndex mv1, MatVarIndex mv2)
  {
    if (mv1.arrayIndex() != mv2.arrayIndex())
      return false;
    return mv1.valueIndex() == mv2.valueIndex();
  }

  //! Comparison operator
  constexpr ARCCORE_HOST_DEVICE friend bool
  operator!=(MatVarIndex mv1, MatVarIndex mv2)
  {
    return !(operator==(mv1, mv2));
  }

  //! Output operator
  ARCANE_CORE_EXPORT friend std::ostream&
  operator<<(std::ostream& o, const MatVarIndex& mvi);

 private:

  Int32 m_array_index;
  Int32 m_value_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Index of a pure material item in a variable.
 */
class PureMatVarIndex
{
 public:

  explicit ARCCORE_HOST_DEVICE PureMatVarIndex(Int32 idx)
  : m_index(idx)
  {}

 public:

  Int32 ARCCORE_HOST_DEVICE valueIndex() const { return m_index; }

 private:

  Int32 m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
