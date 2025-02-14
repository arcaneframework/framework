// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array4View.h                                                (C) 2000-2025 */
/*                                                                           */
/* Vue d'un tableau 4D.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAY4VIEW_H
#define ARCCORE_BASE_ARRAY4VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Array3View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue pour un tableau 4D.
 *
 * Cette classe permet d'obtenir une vue 4D à partir d'une zone contigue
 * en mémoire, comme par exemple celle obtenue via la classe Array.
 *
 * La vue peut s'utiliser comme un tableau C classique, par exemple:
 * \code
 * Array4View<Real> a;
 * a[0][1][2][3] = 5.0;
 * \endcode
 *
 * Il est néammoins préférable d'utiliser directement les méthodes
 * item() ou setItem() (ou l'opérateur operator()) pour accéder en lecture ou
 * écriture à un élément du tableau.
 */
template<class DataType>
class Array4View
{
 public:
  //! Construit une vue
  constexpr Array4View(DataType* ptr,Integer dim1_size,Integer dim2_size,
                       Integer dim3_size,Integer dim4_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size), m_dim3_size(dim3_size),
    m_dim4_size(dim4_size), m_dim34_size(dim3_size*dim4_size),
    m_dim234_size(m_dim34_size*dim2_size)
  {
  }
  //! Construit une vue vide
  constexpr Array4View()
  : m_ptr(0), m_dim1_size(0), m_dim2_size(0), m_dim3_size(0), m_dim4_size(0),
    m_dim34_size(0), m_dim234_size(0)
  {
  }
 public:
  //! Valeur de la première dimension
  constexpr Integer dim1Size() const { return m_dim1_size; }
  //! Valeur de la deuxième dimension
  constexpr Integer dim2Size() const { return m_dim2_size; }
  //! Valeur de la troisième dimension
  constexpr Integer dim3Size() const { return m_dim3_size; }
  //! Valeur de la quatrième dimension
  constexpr Integer dim4Size() const { return m_dim4_size; }
  //! Nombre total d'éléments du tableau
  constexpr Integer totalNbElement() const { return m_dim1_size*m_dim234_size; }
 public:
  constexpr Array3View<DataType> operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return Array3View<DataType>(m_ptr + (m_dim234_size*i),m_dim2_size,m_dim3_size,m_dim4_size);
  }
  constexpr ConstArray3View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArray3View<DataType>(m_ptr + (m_dim234_size*i),m_dim2_size,m_dim3_size,m_dim4_size);
  }
  //! Valeur pour l'élément \a i,j,k,l
  constexpr DataType item(Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
  //! Valeur pour l'élément \a i,j,k,l
  constexpr const DataType& operator()(Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
  //! Valeur pour l'élément \a i,j,k,l
  constexpr DataType& operator()(Integer i,Integer j,Integer k,Integer l)
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Valeur pour l'élément \a i,j,k,l
  constexpr const DataType& operator[](Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
  //! Valeur pour l'élément \a i,j,k,l
  constexpr DataType& operator[](Integer i,Integer j,Integer k,Integer l)
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
#endif
  //! Positionne la valeur pour l'élément \a i,j,k,l
  constexpr void setItem(Integer i,Integer j,Integer k,Integer l,const DataType& value)
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l] = value;
  }
 public:
  /*!
   * \brief Pointeur sur le premier élément du tableau.
   */
  constexpr inline DataType* unguardedBasePointer() { return m_ptr; }
  /*!
   * \brief Pointeur sur le premier élément du tableau.
   */
  constexpr inline DataType* data() { return m_ptr; }
 private:
  DataType* m_ptr;
  Integer m_dim1_size; //!< Taille de la 1ere dimension
  Integer m_dim2_size; //!< Taille de la 2eme dimension
  Integer m_dim3_size; //!< Taille de la 3eme dimension
  Integer m_dim4_size; //!< Taille de la 4eme dimension
  Integer m_dim34_size; //!< dim3 * dim4
  Integer m_dim234_size; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue constante pour un tableau 4D
 */
template<class DataType>
class ConstArray4View
{
 public:
  constexpr ConstArray4View(DataType* ptr,Integer dim1_size,Integer dim2_size,
                            Integer dim3_size,Integer dim4_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size), m_dim3_size(dim3_size),
    m_dim4_size(dim4_size), m_dim34_size(dim3_size*dim4_size),
    m_dim234_size(m_dim34_size*dim2_size)
  {
  }
  constexpr ConstArray4View()
  : m_ptr(nullptr), m_dim1_size(0), m_dim2_size(0), m_dim3_size(0), m_dim4_size(0),
    m_dim34_size(0), m_dim234_size(0)
  {
  }
 public:
  constexpr Integer dim1Size() const { return m_dim1_size; }
  constexpr Integer dim2Size() const { return m_dim2_size; }
  constexpr Integer dim3Size() const { return m_dim3_size; }
  constexpr Integer dim4Size() const { return m_dim4_size; }
  constexpr Integer totalNbElement() const { return m_dim1_size*m_dim234_size; }
 public:
  constexpr ConstArray3View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArray3View<DataType>(m_ptr + (m_dim234_size*i),m_dim2_size,m_dim3_size,m_dim4_size);
  }
  //! Valeur pour l'élément \a i,j,k,l
  constexpr const DataType& operator()(Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Valeur pour l'élément \a i,j,k,l
  constexpr const DataType& operator[](Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
#endif
  constexpr DataType item(Integer i,Integer j,Integer k,Integer l) const
  {
    ARCCORE_CHECK_AT4(i,j,k,l,m_dim1_size,m_dim2_size,m_dim3_size,m_dim4_size);
    return m_ptr[(m_dim234_size*i) + m_dim34_size*j + m_dim4_size*k + l];
  }
 public:

  //! Pointeur sur la mémoire allouée.
  constexpr inline const DataType* unguardedBasePointer() { return m_ptr; }

  //! Pointeur sur la mémoire allouée.
  constexpr inline const DataType* data() { return m_ptr; }

 private:

  const DataType* m_ptr;
  Integer m_dim1_size; //!< Taille de la 1ere dimension
  Integer m_dim2_size; //!< Taille de la 2eme dimension
  Integer m_dim3_size; //!< Taille de la 3eme dimension
  Integer m_dim4_size; //!< Taille de la 4eme dimension
  Integer m_dim34_size; //!< dim3 * dim4
  Integer m_dim234_size; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
