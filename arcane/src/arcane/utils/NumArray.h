// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2021 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAY_H
#define ARCANE_UTILS_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArrayExtents.h"
#include "arcane/utils/ArrayBounds.h"

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
 * \brief Tableaux multi-dimensionnel pour les types numériques sur accélérateur.
 *
 * \warning API en cours de définition.
 */
template<typename DataType,int RankValue>
class MDSpanBase
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,RankValue>;
  // Pour que MDSpan<const T> ait accès à MDSpan<T>
  friend class MDSpanBase<const UnqualifiedValueType,RankValue>;
 public:
  MDSpanBase() = default;
  ARCCORE_HOST_DEVICE MDSpanBase(DataType* ptr,ArrayExtentsWithOffset<RankValue> extents)
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
  ArrayExtentsWithOffset<RankValue> extentsWithOffset() const
  {
    return m_extents;
  }
  Int64 extent(int i) const { return m_extents(i); }
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
  MDSpanBase<const DataType,RankValue> constSpan() const
  { return MDSpanBase<const DataType,RankValue>(m_ptr,m_extents); }
  Span<DataType> to1DSpan() { return { m_ptr, m_extents.totalNbElement() }; }
  Span<const DataType> to1DSpan() const { return { m_ptr, m_extents.totalNbElement() }; }
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
  ArrayExtentsWithOffset<RankValue> m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType,int RankValue>
class MDSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 1D.
template<class DataType>
class MDSpan<DataType,1>
: public MDSpanBase<DataType,1>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,1>;
  using BaseClass = MDSpanBase<DataType,1>;
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
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int64 dim1_size)
  {
    m_extents.setSize(dim1_size);
    m_ptr = ptr;
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<1> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_extents(0); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i) const
  {
    return m_extents.offset(i);
  }
  //! Valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType& operator()(Int64 i) const
  {
    return m_ptr[offset(i)];
  }
  //! Pointeur sur la valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int64 i) const
  {
    return m_ptr+offset(i);
  }
 public:
  MDSpan<const DataType,1> constSpan() const
  { return MDSpan<const DataType,1>(m_ptr,m_extents); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 2D.
template<class DataType>
class MDSpan<DataType,2>
: public MDSpanBase<DataType,2>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,2>;
  using BaseClass = MDSpanBase<DataType,2>;
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
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size)
  {
    m_extents.setSize(dim1_size,dim2_size);
    m_ptr = ptr;
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<2> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int64 dim2Size() const { return m_extents(1); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j) const
  {
    return m_extents.offset(i,j);
  }
  //! Valeur pour l'élément \a i,j
  ARCCORE_HOST_DEVICE DataType& operator()(Int64 i,Int64 j) const
  {
    return m_ptr[offset(i,j)];
  }
  //! Pointeur sur la valeur pour l'élément \a i,j
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int64 i,Int64 j) const
  {
    return m_ptr + offset(i,j);
  }
 public:
  MDSpan<const DataType,2> constSpan() const
  { return MDSpan<const DataType,2>(m_ptr,m_extents); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 3D.
template<class DataType>
class MDSpan<DataType,3>
: public MDSpanBase<DataType,3>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,3>;
  using BaseClass = MDSpanBase<DataType,3>;
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
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size);
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<3> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,UnqualifiedValueType>>>
  ARCCORE_HOST_DEVICE MDSpan(const MDSpan<X,3>& rhs) : BaseClass(rhs){}
 private:
  void _setSize(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size);
    m_ptr = ptr;
  }
 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int64 dim2Size() const { return m_extents(1); }
  //! Valeur de la troisième dimension
  ARCCORE_HOST_DEVICE Int64 dim3Size() const { return m_extents(2); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k) const
  {
    return m_extents.offset(i,j,k);
  }
  //! Valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType& operator()(Int64 i,Int64 j,Int64 k) const
  {
    return m_ptr[offset(i,j,k)];
  }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int64 i,Int64 j,Int64 k) const
  {
    return m_ptr+offset(i,j,k);
  }
 public:
  MDSpan<const DataType,3> constSpan() const
  { return MDSpan<const DataType,3>(m_ptr,m_extents); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 4D.
template<class DataType>
class MDSpan<DataType,4>
: public MDSpanBase<DataType,4>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType,4>;
  using BaseClass = MDSpanBase<DataType,4>;
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
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size,
                             Int64 dim3_size,Int64 dim4_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size,dim4_size);
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<4> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}
 private:
  void _setSize(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size,dim4_size);
    m_ptr = ptr;
  }

 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return m_extents(0); }
  //! Valeur de la deuxième dimension
  Int64 dim2Size() const { return m_extents(1); }
  //! Valeur de la troisième dimension
  Int64 dim3Size() const { return m_extents(2); }
  //! Valeur de la quatrième dimension
  Int64 dim4Size() const { return m_extents(3); }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    return m_extents.offset(i,j,k,l);
  }
 public:
  //! Valeur pour l'élément \a i,j,k,l
  ARCCORE_HOST_DEVICE DataType& operator()(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    return m_ptr[offset(i,j,k,l)];
  }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    return m_ptr + offset(i,j,k,l);
  }
 public:
  MDSpan<const DataType,4> constSpan() const
  { return MDSpan<const DataType,4>(m_ptr,m_extents); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableaux multi-dimensionnel pour les types numériques sur accélérateur.
 *
 * \warning API en cours de définition.
 *
 * On utilise pour l'instant un UniqueArray2 pour conserver les valeurs.
 * La première dimension du UniqueArray2 correspond à extent(0) et la
 * deuxième dimension au dimensions restantes. Par exemple pour un NumArray<Int32,3>
 * ayant comme nombre d'éléments dans chaque dimension (5,9,3), cela
 * correspond à un 'UniqueArray2' dont le nombre d'éléments est (5,9*3).
 */
template<typename DataType,int RankValue>
class NumArrayBase
{
 public:
  using ConstSpanType = MDSpan<const DataType,RankValue>;
  using SpanType = MDSpan<DataType,RankValue>;

 public:
  //! Nombre total d'éléments du tableau
  Int64 totalNbElement() const { return m_total_nb_element; }
  Int64 extent(int i) const { return m_span.extent(i); }
  void resize(ArrayExtents<RankValue> extents)
  {
    m_span.m_extents.setSize(extents);
    _resize();
  }
 protected:
  NumArrayBase() : m_data(platform::getDefaultDataAllocator()){}
  explicit NumArrayBase(ArrayExtents<RankValue> extents)
  : m_data(platform::getDefaultDataAllocator())
  {
    resize(extents);
  }
 private:
  void _resize()
  {
    Int64 dim1_size = extent(0);
    Int64 dim2_size = 1;
    // TODO: vérifier débordement.
    for (int i=1; i<RankValue; ++i )
      dim2_size *= extent(i);
    m_total_nb_element = dim1_size * dim2_size;
    m_data.resize(dim1_size,dim2_size);
    m_span.m_ptr = m_data.to1DSpan().data();
  }
 public:
  void fill(const DataType& v) { m_data.fill(v); }
  Int32 nbDimension() const { return RankValue; }
  ArrayExtents<RankValue> extents() const { return m_span.extents(); }
  ArrayExtentsWithOffset<RankValue> extentsWithOffset() const
  {
    return m_span.extentsWithOffset();
  }
 public:
  SpanType span() { return m_span; }
  ConstSpanType span() const { return m_span.constSpan(); }
  ConstSpanType constSpan() const { return m_span.constSpan(); }
 public:
  Span<const DataType> to1DSpan() const { return m_data.to1DSpan(); }
  Span<DataType> to1DSpan() { return m_data.to1DSpan(); }
  void copy(ConstSpanType rhs) { m_data.copy(rhs._internalTo2DSpan()); }
  const DataType& operator()(ArrayBoundsIndex<RankValue> idx) const
  {
    return m_span(idx);
  }
  DataType& s(ArrayBoundsIndex<RankValue> idx)
  {
    return m_span(idx);
  }
  void swap(NumArrayBase<DataType,RankValue>& rhs)
  {
    m_data.swap(rhs.m_data);
    std::swap(m_span,rhs.m_span);
    std::swap(m_total_nb_element,rhs.m_total_nb_element);
  }
  Int64 capacity() const { return m_data.capacity(); }
 public:
  //! \internal
  DataType* _internalData() { return m_span._internalData(); }
 protected:
  MDSpan<DataType,RankValue> m_span;
  UniqueArray2<DataType> m_data;
  Int64 m_total_nb_element = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 1D.
template<class DataType>
class NumArray<DataType,1>
: public NumArrayBase<DataType,1>
{
 public:
  using BaseClass = NumArrayBase<DataType,1>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0){}
  //! Construit un tableau
  NumArray(Int64 dim1_size) : BaseClass(ArrayExtents<1>(dim1_size)){}
 public:
  void resize(Int64 dim1_size)
  {
    this->resize(ArrayExtents<1>(dim1_size));
  }
 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return this->extent(0); }
 public:
  //! Valeur pour l'élément \a i
  DataType operator()(Int64 i) const
  {
    return m_span(i);
  }
  //! Positionne la valeur pour l'élément \a i
  DataType& s(Int64 i)
  {
    return m_span(i);
  }
 public:
  operator MDSpan<DataType,1> () { return this->span(); }
  operator MDSpan<const DataType,1> () const { return this->constSpan(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 2D.
template<class DataType>
class NumArray<DataType,2>
: public NumArrayBase<DataType,2>
{
 public:
  using BaseClass = NumArrayBase<DataType,2>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size)
  : BaseClass(ArrayExtents<2>(dim1_size,dim2_size)){}
 public:
  void resize(Int64 dim1_size,Int64 dim2_size)
  {
    this->resize(ArrayExtents<2>(dim1_size,dim2_size));
  }

 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  Int64 dim2Size() const { return extent(1); }
 public:
  //! Valeur pour l'élément \a i,j
  DataType operator()(Int64 i,Int64 j) const
  {
    return m_span(i,j);
  }
  //! Positionne la valeur pour l'élément \a i,j
  DataType& s(Int64 i,Int64 j)
  {
    return m_span(i,j);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 3D.
template<class DataType>
class NumArray<DataType,3>
: public NumArrayBase<DataType,3>
{
 public:
  using BaseClass = NumArrayBase<DataType,3>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  : BaseClass(ArrayExtents<3>(dim1_size,dim2_size,dim3_size)){}
 public:
  void resize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    this->resize(ArrayExtents<3>(dim1_size,dim2_size,dim3_size));
  }
 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  Int64 dim2Size() const { return extent(1); }
  //! Valeur de la troisième dimension
  Int64 dim3Size() const { return extent(2); }
 public:
  //! Valeur pour l'élément \a i,j,k
  DataType operator()(Int64 i,Int64 j,Int64 k) const
  {
    return m_span(i,j,k);
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& s(Int64 i,Int64 j,Int64 k)
  {
    return m_span(i,j,k);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 4D.
template<class DataType>
class NumArray<DataType,4>
: public NumArrayBase<DataType,4>
{
 public:
  using BaseClass = NumArrayBase<DataType,4>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size,
           Int64 dim3_size,Int64 dim4_size)
  : BaseClass(ArrayExtents<4>(dim1_size,dim2_size,dim3_size,dim4_size)){}
 public:
  void resize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    this->resize(ArrayExtents<4>(dim1_size,dim2_size,dim3_size,dim4_size));
  }

 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  Int64 dim2Size() const { return extent(1); }
  //! Valeur de la troisième dimension
  Int64 dim3Size() const { return extent(2); }
  //! Valeur de la quatrième dimension
  Int64 dim4Size() const { return extent(3); }
 public:
  //! Valeur pour l'élément \a i,j,k,l
  DataType operator()(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    return m_span(i,j,k,l);
  }
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& s(Int64 i,Int64 j,Int64 k,Integer l)
  {
    return m_span(i,j,k,l);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
