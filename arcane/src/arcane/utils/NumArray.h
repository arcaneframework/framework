// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2021 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques sur accélérateur.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAY_H
#define ARCANE_UTILS_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

template<typename DataType,int RankValue>
class NumArrayBase;

namespace impl
{
template <class T> ARCCORE_HOST_DEVICE
T fastmod(T a , T b)
{
  return a < b ? a : a-b*(a/b);
}
}

template<int RankValue> class ArrayBounds;
template<int RankValue> class ArrayBoundsIndex;
template<int RankValue> class ArrayExtents;
template<int RankValue> class ArrayExtentsWithOffset;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBoundsIndex<1>
{
 public:
  ARCCORE_HOST_DEVICE ArrayBoundsIndex(Int64 _id0)
  {
    id0 = _id0;
  }
 public:
  Int64 id0;
};

template<>
class ArrayBoundsIndex<2>
{
 public:
  ARCCORE_HOST_DEVICE ArrayBoundsIndex(Int64 _id0,Int64 _id1)
  {
    id0 = _id0;
    id1 = _id1;
  }
 public:
  Int64 id0;
  Int64 id1;
};

template<>
class ArrayBoundsIndex<3>
{
 public:
  ARCCORE_HOST_DEVICE ArrayBoundsIndex(Int64 _id0,Int64 _id1,Int64 _id2)
  {
    id0 = _id0;
    id1 = _id1;
    id2 = _id2;
  }
 public:
  Int64 id0;
  Int64 id1;
  Int64 id2;
};

template<>
class ArrayBoundsIndex<4>
{
 public:
  ARCCORE_HOST_DEVICE ArrayBoundsIndex(Int64 _id0,Int64 _id1,Int64 _id2,Int64 _id3)
  {
    id0 = _id0;
    id1 = _id1;
    id2 = _id2;
    id3 = _id3;
  }
 public:
  Int64 id0;
  Int64 id1;
  Int64 id2;
  Int64 id3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour conserver le nombre d'éléments dans chaque dimension.
 */
template<int RankValue>
class ArrayExtentsBase
{
 public:
  ARCCORE_HOST_DEVICE ArrayExtentsBase()
  {
    for( int i=0; i<RankValue; ++i )
      m_extents[i] = 0;
  }
  //! Nombre d'élément de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE Int64 extent(int i) const { return m_extents[i]; }
  ARCCORE_HOST_DEVICE Int64 operator()(int i) const { return m_extents[i]; }
  ARCCORE_HOST_DEVICE Span<const Int64> extentsAsSpan() const { return { m_extents, RankValue }; }
  ARCCORE_HOST_DEVICE Int64 totalNbElement() const
  {
    Int64 nb_element = 1;
    for (int i=0; i<RankValue; i++)
      nb_element *= m_extents[i];
    return nb_element;
  }
 protected:
  Int64 m_extents[RankValue];
};

template<>
class ArrayExtents<1>
: public ArrayExtentsBase<1>
{
 public:
  ArrayExtents() = default;
  ARCCORE_HOST_DEVICE explicit ArrayExtents(Int64 dim1_size)
  {
    setSize(dim1_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size)
  {
    m_extents[0] = dim1_size;
  }
};

template<>
class ArrayExtents<2>
: public ArrayExtentsBase<2>
{
 public:
  ArrayExtents() = default;
  ARCCORE_HOST_DEVICE ArrayExtents(Int64 dim1_size,Int64 dim2_size)
  {
    setSize(dim1_size,dim2_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
  }
};

template<>
class ArrayExtents<3>
: public ArrayExtentsBase<3>
{
 public:
  ArrayExtents() = default;
  ARCCORE_HOST_DEVICE ArrayExtents(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    setSize(dim1_size,dim2_size,dim3_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
  }
};

template<>
class ArrayExtents<4>
: public ArrayExtentsBase<4>
{
 public:
  ArrayExtents() = default;
  ARCCORE_HOST_DEVICE ArrayExtents(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    setSize(dim1_size,dim2_size,dim3_size,dim4_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
    m_extents[3] = dim4_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<1>
: private ArrayExtents<1>
{
 public:
  using BaseClass = ArrayExtents<1>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::extentsAsSpan;
  using BaseClass::totalNbElement;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<1> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    return i;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<1> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0,m_extents[0]);
    return idx.id0;
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size)
  {
    BaseClass::setSize(dim1_size);
  }
  ARCCORE_HOST_DEVICE void setSize(ArrayExtents<1> extents)
  {
    BaseClass::setSize(extents(0));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<2>
: private ArrayExtents<2>
{
 public:
  using BaseClass = ArrayExtents<2>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::extentsAsSpan;
  using BaseClass::totalNbElement;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<2> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    return m_extents[1]*i + j;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<3> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0,m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1,m_extents[1]);
    return m_extents[1]*idx.id0 + idx.id1;
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size)
  {
    BaseClass::setSize(dim1_size,dim2_size);
  }
  ARCCORE_HOST_DEVICE void setSize(ArrayExtents<2> dims)
  {
    this->setSize(dims(0),dims(1));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<3>
: private ArrayExtents<3>
{
 public:
  using BaseClass = ArrayExtents<3>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::extentsAsSpan;
  using BaseClass::totalNbElement;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<3> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    ARCCORE_CHECK_AT(k,m_extents[2]);
    return (m_dim23_size*i) + m_extents[2]*j + k;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<3> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0,m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1,m_extents[1]);
    ARCCORE_CHECK_AT(idx.id2,m_extents[2]);
    return (m_dim23_size*idx.id0) + m_extents[2]*idx.id1 + idx.id2;
  }
  void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    BaseClass::setSize(dim1_size,dim2_size,dim3_size);
    _computeOffsets();
  }
  void setSize(ArrayExtents<3> dims)
  {
    this->setSize(dims(0),dims(1),dims(2));
  }
 protected:
  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    m_dim23_size = m_extents[1] * m_extents[2];
  }
 private:
  Int64 m_dim23_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<4>
: private ArrayExtents<4>
{
 public:
  using BaseClass = ArrayExtents<4>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::extentsAsSpan;
  using BaseClass::totalNbElement;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<4> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    ARCCORE_CHECK_AT(k,m_extents[2]);
    ARCCORE_CHECK_AT(l,m_extents[3]);
    return (m_dim234_size*i) + m_dim34_size*j + m_extents[3]*k + l;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<4> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0,m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1,m_extents[1]);
    ARCCORE_CHECK_AT(idx.id2,m_extents[2]);
    ARCCORE_CHECK_AT(idx.id3,m_extents[3]);
    return (m_dim234_size*idx.id0) + m_dim34_size*idx.id1 + m_extents[3]*idx.id2 + idx.id3;
  }
  void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    BaseClass::setSize(dim1_size,dim2_size,dim3_size,dim4_size);
    _computeOffsets();
  }
  void setSize(ArrayExtents<4> dims)
  {
    this->setSize(dims(0),dims(1),dims(2),dims(3));
  }
 protected:
  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    m_dim34_size = m_extents[2] * m_extents[3];
    m_dim234_size = m_dim34_size * m_extents[1];
  }
 private:
  Int64 m_dim34_size = 0; //!< dim3 * dim4
  Int64 m_dim234_size = 0; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int RankValue>
class ArrayBoundsBase
: private ArrayExtents<RankValue>
{
 public:
  using ArrayExtents<RankValue>::extent;
 public:
  ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_nb_element; }
 protected:
  void _computeNbElement()
  {
    m_nb_element = this->totalNbElement();
  }
 protected:
  using ArrayExtents<RankValue>::m_extents;
  Int64 m_nb_element = 0;
};

template<>
class ArrayBounds<1>
: public ArrayBoundsBase<1>
{
 public:
  using IndexType = ArrayBoundsIndex<1>;
  using ArrayBoundsBase<1>::m_extents;
  explicit ArrayBounds(Int64 dim1)
  {
    m_extents[0] = dim1;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE IndexType getIndices(Int64 i) const
  {
    return { i };
  }
};

template<>
class ArrayBounds<2>
: public ArrayBoundsBase<2>
{
 public:
  using IndexType = ArrayBoundsIndex<2>;
  using ArrayBoundsBase<2>::m_extents;
  ArrayBounds(Int64 dim1,Int64 dim2)
  {
    m_extents[0] = dim1;
    m_extents[1] = dim2;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE IndexType getIndices(Int64 i) const
  {
    Int64 i1 = impl::fastmod(i,m_extents[1]);
    Int64 i0 = i / m_extents[1];
    return { i0, i1 };
  }
};

template<>
class ArrayBounds<3>
: public ArrayBoundsBase<3>
{
 public:
  using IndexType = ArrayBoundsIndex<3>;
  using ArrayBoundsBase<3>::m_extents;
  ArrayBounds(Int64 dim1,Int64 dim2,Int64 dim3)
  {
    m_extents[0] = dim1;
    m_extents[1] = dim2;
    m_extents[2] = dim3;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE IndexType getIndices(Int64 i) const
  {
    Int64 i2 = impl::fastmod(i,m_extents[2]);
    Int64 fac = m_extents[2];
    Int64 i1 = impl::fastmod(i / fac,m_extents[1]);
    fac *= m_extents[1];
    Int64 i0 = i / fac;
    return { i0, i1, i2 };
  }
};

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
 public:
  MDSpanBase() = default;
  ARCCORE_HOST_DEVICE MDSpanBase(DataType* ptr,ArrayExtentsWithOffset<RankValue> extents)
  : m_ptr(ptr), m_extents(extents)
  {
  }
 public:
  ARCCORE_HOST_DEVICE DataType* _internalData() { return m_ptr; }
  ARCCORE_HOST_DEVICE const DataType* _internalData() const { return m_ptr; }
 public:
  SmallSpan<const Int64> extents() const { return m_extents.extents(); }
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
  friend class NumArrayBase<DataType,1>;
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
  friend class NumArrayBase<DataType,2>;
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
  friend class NumArrayBase<DataType,3>;
  using BaseClass = MDSpanBase<DataType,3>;
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
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size);
  }
  ARCCORE_HOST_DEVICE MDSpan(DataType* ptr,ArrayExtentsWithOffset<3> extents_and_offset)
  : BaseClass(ptr,extents_and_offset) {}
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
  friend class NumArrayBase<DataType,4>;
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
 */
template<typename DataType,int RankValue>
class NumArrayBase
{
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
    Int64 full_size = extent(0);
    // TODO: vérifier débordement.
    for (int i=1; i<RankValue; ++i )
      full_size *= extent(i);
    m_total_nb_element = full_size;
    m_data.resize(full_size);
    m_span.m_ptr = m_data.data();
  }
 public:
  void fill(const DataType& v) { m_data.fill(v); }
  DataType* _internalData() { return m_span._internalData(); }
  Int32 nbDimension() const { return RankValue; }
  SmallSpan<const Int64> extents() const { return m_span.extents(); }
 public:
  MDSpan<DataType,RankValue> span() { return m_span; }
  MDSpan<const DataType,RankValue> span() const { return m_span.constSpan(); }
  MDSpan<const DataType,RankValue> constSpan() const { return m_span.constSpan(); }
 protected:
  MDSpan<DataType,RankValue> m_span;
  UniqueArray<DataType> m_data;
  Int64 m_total_nb_element = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType,int RankValue>
class NumArray;

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
