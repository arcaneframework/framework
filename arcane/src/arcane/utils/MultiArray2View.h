// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2View.h                                           (C) 2000-2025 */
/*                                                                           */
/* Vue d'un tableau 2D à taille multiple.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIARRAY2VIEW_H
#define ARCANE_UTILS_MULTIARRAY2VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ArrayView.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue modifiable sur un MultiArray2.
 */
template <class DataType>
class MultiArray2View
{
 public:

  //! Vue sur la tableau \a buf
  MultiArray2View(ArrayView<DataType> buf, ConstArrayView<Int32> indexes, ConstArrayView<Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}
  //! Vue vide
  MultiArray2View() = default;

 public:

  //! Nombre d'éléments de la première dimension.
  Int32 dim1Size() const { return m_sizes.size(); }
  /*!
   * \brief Nombre d'éléments de la première dimension.
   * \deprecated Utiliser dim1Size() à la place.
   */
  ARCANE_DEPRECATED_122 Int32 size() const { return dim1Size(); }
  //! Nombre d'éléments de la deuxième dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }
  //! Nombre total d'éléments dans le tableau.
  Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! \a i-ème élément du tableau
  ArrayView<DataType> operator[](Int32 i)
  {
    return ArrayView<DataType>(this->m_sizes[i], &this->m_buffer[this->m_indexes[i]]);
  }
  //! \a i-ème élément du tableau
  ConstArrayView<DataType> operator[](Int32 i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i], this->m_buffer.data() + (this->m_indexes[i]));
  }

 private:

  ArrayView<DataType> m_buffer;
  ConstArrayView<Int32> m_indexes;
  ConstArrayView<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue constante sur un MultiArray2.
 */
template <class DataType>
class ConstMultiArray2View
{
 private:

  friend class MultiArray2<DataType>;

 public:

  //! Vue sur la tableau \a buf
  ConstMultiArray2View(ConstArrayView<DataType> buf, ConstArrayView<Int32> indexes,
                       ConstArrayView<Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}
  //! Vue vide
  ConstMultiArray2View() = default;

 public:

  //! Nombre d'éléments de la première dimension.
  Int32 dim1Size() const { return m_sizes.size(); }
  /*!
   * \brief Nombre d'éléments de la première dimension.
   * \deprecated Utiliser dim1Size() à la place.
   */
  ARCANE_DEPRECATED_122 Int32 size() const { return dim1Size(); }
  //! Nombre d'éléments de la deuxième dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }
  //! Nombre total d'éléments dans le tableau.
  Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! \a i-ème élément du tableau
  ConstArrayView<DataType> operator[](Int32 i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i], this->m_buffer.data() + (this->m_indexes[i]));
  }

 private:

  ConstArrayView<DataType> m_buffer;
  ConstArrayView<Int32> m_indexes;
  ConstArrayView<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un MultiArray2.
 *
 * Les instances de cette classe sont créées par appel à MultiArray2::span()
 * ou MultiArray2::constSpan().
 */
template <class DataType>
class JaggedSmallSpan
{
 private:

  friend class MultiArray2<std::remove_cv_t<DataType>>;

 public:

  //! Vue vide
  JaggedSmallSpan() = default;

 private:

  //! Vue sur la tableau \a buf
  JaggedSmallSpan(SmallSpan<DataType> buf, SmallSpan<const Int32> indexes,
                       SmallSpan<const Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}

 public:

  //! Nombre d'éléments de la première dimension.
  constexpr ARCCORE_HOST_DEVICE Int32 dim1Size() const { return m_sizes.size(); }
  //! Nombre d'éléments de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const Int32> dim2Sizes() const { return m_sizes; }
  //! Nombre total d'éléments dans le tableau.
  constexpr ARCCORE_HOST_DEVICE Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! \a i-ème élément du tableau
  constexpr ARCCORE_HOST_DEVICE SmallSpan<DataType> operator[](Int32 i) const
  {
    return m_buffer.subSpan(m_indexes[i], m_sizes[i]);
  }

 private:

  SmallSpan<DataType> m_buffer;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
