// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2View.h                                           (C) 2000-2015 */
/*                                                                           */
/* Vue d'un tableau 2D à taille multiple.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIARRAY2VIEW_H
#define ARCANE_UTILS_MULTIARRAY2VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> class MultiArray2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue modifiable sur un MultiArray2.
 */
template<class DataType>
class MultiArray2View
{
 public:
  //! Vue sur la tableau \a buf
  MultiArray2View(ArrayView<DataType> buf,IntegerConstArrayView indexes,IntegerConstArrayView sizes)
  : m_buffer(buf), m_indexes(indexes), m_sizes(sizes) { }
  //! Vue vide
  MultiArray2View() { }
 public:
  //! Nombre d'éléments de la première dimension.
  Integer dim1Size() const { return m_sizes.size(); }
  /*!
   * \brief Nombre d'éléments de la première dimension.
   * \deprecated Utiliser dim1Size() à la place.
   */
  ARCANE_DEPRECATED_122 Integer size() const { return dim1Size(); }
  //! Nombre d'éléments de la deuxième dimension
  IntegerConstArrayView dim2Sizes() const { return m_sizes; }
  //! Nombre total d'éléments dans le tableau.
  Integer totalNbElement() const { return m_buffer.size(); }
 public:
  //! \a i-ème élément du tableau
  ArrayView<DataType> operator[](Integer i)
    {
      return ArrayView<DataType>(this->m_sizes[i],&this->m_buffer[this->m_indexes[i]]);
    }
  //! \a i-ème élément du tableau
  ConstArrayView<DataType> operator[](Integer i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i],this->m_buffer.data()+ (this->m_indexes[i]));
  }
 private:
  ArrayView<DataType> m_buffer;
  IntegerConstArrayView m_indexes;
  IntegerConstArrayView m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue constante sur un MultiArray2.
 */
template<class DataType>
class ConstMultiArray2View
{
 private:
  friend class MultiArray2<DataType>;
 public:
  //! Vue sur la tableau \a buf
  ConstMultiArray2View(ConstArrayView<DataType> buf,IntegerConstArrayView indexes,
                       IntegerConstArrayView sizes)
  : m_buffer(buf), m_indexes(indexes), m_sizes(sizes) { }
  //! Vue vide
  ConstMultiArray2View() { }
 public:
  //! Nombre d'éléments de la première dimension.
  Integer dim1Size() const { return m_sizes.size(); }
  /*!
   * \brief Nombre d'éléments de la première dimension.
   * \deprecated Utiliser dim1Size() à la place.
   */
  ARCANE_DEPRECATED_122 Integer size() const { return dim1Size(); }
  //! Nombre d'éléments de la deuxième dimension
  IntegerConstArrayView dim2Sizes() const { return m_sizes; }
  //! Nombre total d'éléments dans le tableau.
  Integer totalNbElement() const { return m_buffer.size(); }
 public:
  //! \a i-ème élément du tableau
  ConstArrayView<DataType> operator[](Integer i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i],this->m_buffer.data()+ (this->m_indexes[i]));
  }
 private:
  ConstArrayView<DataType> m_buffer;
  IntegerConstArrayView m_indexes;
  IntegerConstArrayView m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
