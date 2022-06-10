// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDSpan.h                                                    (C) 2000-2022 */
/*                                                                           */
/* Vue sur un tableaux multi-dimensionnel pour les types numériques.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MDSPAN_H
#define ARCANE_UTILS_MDSPAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayExtents.h"
#include "arcane/utils/ArrayBounds.h"
#include "arcane/utils/NumericTraits.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues multi-dimensionnelles.
 *
 * \warning API en cours de définition.
 *
 * Cette classe s'inspire la classe std::mdspan en cours de définition
 * (voir http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0009r12.html)
 *
 * Cette classe est utilisée pour gérer les vues sur les tableaux des types
 * numériques qui sont accessibles sur accélérateur. \a RankValue est le
 * rang du tableau (nombre de dimensions) et \a DataType le type de données
 * associé.
 *
 * En général cette classe n'est pas utilisée directement mais par l'intermédiaire
 * d'une de ses spécialisations suivant le rang comme MDSpan<DataType,1>,
 * MDSpan<DataType,2>, MDSpan<DataType,3> ou MDSpan<DataType,4>.
 */
template<typename DataType,int RankValue,typename LayoutType>
class MDSpanBase
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,RankValue,LayoutType>;
  // Pour que MDSpan<const T> ait accès à MDSpan<T>
  friend class MDSpanBase<const UnqualifiedValueType,RankValue,LayoutType>;
 public:
  MDSpanBase() = default;
  ARCCORE_HOST_DEVICE MDSpanBase(DataType* ptr,ArrayExtentsWithOffset<RankValue,LayoutType> extents)
  : m_ptr(ptr), m_extents(extents)
  {
  }
  // Constructeur MDSpan<const T> à partir d'un MDSpan<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,UnqualifiedValueType>>>
  ARCCORE_HOST_DEVICE MDSpanBase(const MDSpanBase<X,RankValue>& rhs)
  : m_ptr(rhs.m_ptr), m_extents(rhs.m_extents){}
 public:
  ARCCORE_HOST_DEVICE DataType* _internalData() { return m_ptr; }
  ARCCORE_HOST_DEVICE const DataType* _internalData() const { return m_ptr; }
 public:
  ArrayExtents<RankValue> extents() const
  {
    return m_extents.extents();
  }
  ArrayExtentsWithOffset<RankValue,LayoutType> extentsWithOffset() const
  {
    return m_extents;
  }
  Int32 extent(int i) const { return m_extents(i); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<RankValue> idx) const
  {
    return m_extents.offset(idx);
  }
  //! Valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType& operator()(ArrayBoundsIndex<RankValue> idx) const
  {
    return m_ptr[offset(idx)];
  }
  //! Pointeur sur la valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType* ptrAt(ArrayBoundsIndex<RankValue> idx) const
  {
    return m_ptr+offset(idx);
  }
 public:
  ARCCORE_HOST_DEVICE MDSpanBase<const DataType,RankValue> constSpan() const
  { return MDSpanBase<const DataType,RankValue>(m_ptr,m_extents); }
  ARCCORE_HOST_DEVICE Span<DataType> to1DSpan() const { return { m_ptr, m_extents.totalNbElement() }; }
 private:
  // Utilisé uniquement par NumArrayBase pour la copie
  Span2<const DataType> _internalTo2DSpan() const
  {
    Int64 dim1_size = m_extents(0);
    Int64 dim2_size = 1;
    for (int i=1; i<RankValue; ++i )
      dim2_size *= m_extents(i);
    return { m_ptr, dim1_size, dim2_size };
  }
 protected:
  DataType* m_ptr = nullptr;
  ArrayExtentsWithOffset<RankValue,LayoutType> m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue multi-dimensionnelle à 1 dimension.
 */
template<class DataType,typename LayoutType>
class MDSpan<DataType,1,LayoutType>
: public MDSpanBase<DataType,1,LayoutType>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,1,LayoutType>;
  using BaseClass = MDSpanBase<DataType,1,LayoutType>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
 public:
  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
 public:
  //! Construit un tableau vide
  MDSpan() = default;
  //! Construit un tableau
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int32 dim1_size)
  {
    m_extents.setSize(dim1_size);
    m_ptr = ptr;
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<1,LayoutType> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_extents(0); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i) const { return m_extents.offset(i); }
  //! Valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i) const { return m_ptr[offset(i)]; }
  //! Pointeur sur la valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i) const { return m_ptr+offset(i); }
  //! Valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType operator[](Int32 i) const { return m_ptr[offset(i)]; }
  //! Valeur pour l'élément \a i et la composante \a a
  template<typename X = DataType,typename SubType = typename NumericTraitsT<X>::SubscriptType >
  ARCCORE_HOST_DEVICE SubType operator()(Int32 i,Int32 a) const { return m_ptr[offset(i)][a]; }
  //! Valeur pour l'élément \a i et la composante \a [a][b]
  template<typename X = DataType,typename Sub2Type = typename NumericTraitsT<X>::Subscript2Type >
  ARCCORE_HOST_DEVICE Sub2Type operator()(Int32 i,Int32 a,Int32 b) const { return m_ptr[offset(i)][a][b]; }
 public:
  ARCCORE_HOST_DEVICE MDSpan<const DataType,1,LayoutType> constSpan() const
  { return MDSpan<const DataType,1,LayoutType>(m_ptr,m_extents); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue multi-dimensionnelle à 2 dimensions.
 */
template<class DataType,typename LayoutType>
class MDSpan<DataType,2,LayoutType>
: public MDSpanBase<DataType,2,LayoutType>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,2,LayoutType>;
  using BaseClass = MDSpanBase<DataType,2,LayoutType>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
 public:
  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
 public:
  //! Construit un tableau vide
  MDSpan() : MDSpan(nullptr,0,0){}
  //! Construit une vue
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int32 dim1_size,Int32 dim2_size)
  {
    m_extents.setSize(dim1_size,dim2_size);
    m_ptr = ptr;
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<2,LayoutType> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int32 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int32 dim2Size() const { return m_extents(1); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j) const { return m_extents.offset(i,j); }
  //! Valeur pour l'élément \a i,j
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i,Int32 j) const { return m_ptr[offset(i,j)]; }
  //! Pointeur sur la valeur pour l'élément \a i,j
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i,Int32 j) const { return m_ptr + offset(i,j); }
  //! Valeur pour l'élément \a i et la composante \a a
  template<typename X = DataType,typename SubType = typename NumericTraitsT<X>::SubscriptType >
  ARCCORE_HOST_DEVICE SubType operator()(Int32 i,Int32 j,Int32 a) const { return m_ptr[offset(i,j)][a]; }
  //! Valeur pour l'élément \a i et la composante \a [a][b]
  template<typename X = DataType,typename Sub2Type = typename NumericTraitsT<X>::Subscript2Type >
  ARCCORE_HOST_DEVICE Sub2Type operator()(Int32 i,Int32 j,Int32 a,Int32 b) const { return m_ptr[offset(i,j)][a][b]; }
 public:
  ARCCORE_HOST_DEVICE MDSpan<const DataType,2,LayoutType> constSpan() const
  {
    return MDSpan<const DataType,2,LayoutType>(m_ptr,m_extents);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue multi-dimensionnelle à 3 dimensions.
 */
template<class DataType,typename LayoutType>
class MDSpan<DataType,3,LayoutType>
: public MDSpanBase<DataType,3,LayoutType>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,3,LayoutType>;
  using BaseClass = MDSpanBase<DataType,3,LayoutType>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
  using value_type = typename std::remove_cv<DataType>::type;
 public:
  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
 public:
  //! Construit un tableau vide
  MDSpan() = default;
  //! Construit une vue
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size);
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<3,LayoutType> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,UnqualifiedValueType>>>
  ARCCORE_HOST_DEVICE MDSpan(const MDSpan<X,3>& rhs) : BaseClass(rhs){}
 private:
  void _setSize(DataType* ptr,Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size);
    m_ptr = ptr;
  }
 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int32 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int32 dim2Size() const { return m_extents(1); }
  //! Valeur de la troisième dimension
  ARCCORE_HOST_DEVICE Int32 dim3Size() const { return m_extents(2); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j,Int32 k) const { return m_extents.offset(i,j,k); }
  //! Valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i,Int32 j,Int32 k) const { return m_ptr[offset(i,j,k)]; }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i,Int32 j,Int32 k) const { return m_ptr+offset(i,j,k); }
  //! Valeur pour l'élément \a i et la composante \a a
  template<typename X = DataType,typename SubType = typename NumericTraitsT<X>::SubscriptType >
  ARCCORE_HOST_DEVICE SubType operator()(Int32 i,Int32 j,Int32 k,Int32 a) const
  {
    return m_ptr[offset(i,j,k)][a];
  }
  //! Valeur pour l'élément \a i et la composante \a [a][b]
  template<typename X = DataType,typename Sub2Type = typename NumericTraitsT<X>::Subscript2Type >
  ARCCORE_HOST_DEVICE Sub2Type operator()(Int32 i,Int32 j,Int32 k,Int32 a,Int32 b) const
  {
    return m_ptr[offset(i,j,k)][a][b];
  }
 public:
  ARCCORE_HOST_DEVICE MDSpan<const DataType,3,LayoutType> constSpan() const
  {
    return MDSpan<const DataType,3,LayoutType>(m_ptr,m_extents);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue multi-dimensionnelle à 4 dimensions.
 */
template<class DataType,typename LayoutType>
class MDSpan<DataType,4,LayoutType>
: public MDSpanBase<DataType,4,LayoutType>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,4,LayoutType>;
  using BaseClass = MDSpanBase<DataType,4,LayoutType>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
 public:
  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
 public:
  //! Construit un tableau vide
  MDSpan() = default;
  //! Construit une vue
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int32 dim1_size,Int32 dim2_size,
                             Int32 dim3_size,Int32 dim4_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size,dim4_size);
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<4,LayoutType> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}
 private:
  void _setSize(DataType* ptr,Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,Int32 dim4_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size,dim4_size);
    m_ptr = ptr;
  }

 public:
  //! Valeur de la première dimension
  Int32 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  Int32 dim2Size() const { return m_extents(1); }
  //! Valeur de la troisième dimension
  Int32 dim3Size() const { return m_extents(2); }
  //! Valeur de la quatrième dimension
  Int32 dim4Size() const { return m_extents(3); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return m_extents.offset(i,j,k,l);
  }
 public:
  //! Valeur pour l'élément \a i,j,k,l
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return m_ptr[offset(i,j,k,l)];
  }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return m_ptr + offset(i,j,k,l);
  }
  //! Valeur pour l'élément \a i et la composante \a a
  template<typename X = DataType,typename SubType = typename NumericTraitsT<X>::SubscriptType >
  ARCCORE_HOST_DEVICE SubType operator()(Int32 i,Int32 j,Int32 k,Int32 l,Int32 a) const
  {
    return m_ptr[offset(i,j,k,l)][a];
  }
  //! Valeur pour l'élément \a i et la composante \a [a][b]
  template<typename X = DataType,typename Sub2Type = typename NumericTraitsT<X>::Subscript2Type >
  ARCCORE_HOST_DEVICE Sub2Type operator()(Int32 i,Int32 j,Int32 k,Int32 l,Int32 a,Int32 b) const
  {
    return m_ptr[offset(i,j,k,l)][a][b];
  }
 public:
  ARCCORE_HOST_DEVICE MDSpan<const DataType,4,LayoutType> constSpan() const
  {
    return MDSpan<const DataType,4,LayoutType>(m_ptr,m_extents);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
