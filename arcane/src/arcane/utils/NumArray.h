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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour conserver le nombre d'éléments dans chaque dimension.
 */
template<int RankValue>
class ArrayExtentsBase
{
 public:
  //! Nombre d'élément de la \a i-ème dimension.
  Int64 extent(int i) const { return m_extents[i]; }
  Span<const Int64> extents() const { return { m_extents, RankValue }; }
 protected:
  Int64 m_extents[RankValue];
};

template<>
class ArrayExtents<1>
: public ArrayExtentsBase<1>
{
};

template<>
class ArrayExtents<2>
: public ArrayExtentsBase<2>
{
};

template<>
class ArrayExtents<3>
: public ArrayExtentsBase<3>
{
};

template<>
class ArrayExtents<4>
: public ArrayExtentsBase<4>
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<1>
: public ArrayExtents<1>
{
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    return i;
  }
  void setSize(Int64 dim1_size)
  {
    m_extents[0] = dim1_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<2>
: public ArrayExtents<2>
{
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    return m_extents[1]*i + j;
  }
  void setSize(Int64 dim1_size,Int64 dim2_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<3>
: public ArrayExtents<3>
{
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    ARCCORE_CHECK_AT(k,m_extents[2]);
    return (m_dim23_size*i) + m_extents[2]*j + k;
  }
  void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    m_dim23_size = dim2_size * dim3_size;

    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
  }
 private:
  Int64 m_dim23_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtentsWithOffset<4>
: public ArrayExtents<4>
{
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    ARCCORE_CHECK_AT(i,m_extents[0]);
    ARCCORE_CHECK_AT(j,m_extents[1]);
    ARCCORE_CHECK_AT(k,m_extents[2]);
    ARCCORE_CHECK_AT(l,m_extents[3]);
    return (m_dim234_size*i) + m_dim34_size*j + m_extents[3]*k + l;
  }
  void setSize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    m_dim34_size = dim3_size*dim4_size;
    m_dim234_size = m_dim34_size*dim2_size;
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
    m_extents[3] = dim4_size;
  }
 private:
  Int64 m_dim34_size = 0; //!< dim3 * dim4
  Int64 m_dim234_size = 0; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int RankValue>
class ArrayBoundsBase
: protected ArrayExtents<RankValue>
{
 public:
  using ArrayExtents<RankValue>::extent;
 public:
  ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_nb_element; }
 protected:
  void _computeNbElement()
  {
    m_nb_element = 1;
    for (int i=0; i<RankValue; i++)
      m_nb_element *= this->m_extents[i];
  }
 protected:
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
  ARCCORE_HOST_DEVICE DataType* _internalData() { return m_ptr; }
  ARCCORE_HOST_DEVICE const DataType* _internalData() const { return m_ptr; }
 protected:
  DataType* m_ptr = nullptr;
  Int64 m_dimensions[RankValue];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType,int rank>
class MDSpan;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 1D.
template<class DataType>
class MDSpan<DataType,1>
: public MDSpanBase<DataType,1>
{
  using BaseClass = MDSpanBase<DataType,1>;
  using BaseClass::m_dimensions;
  using BaseClass::m_ptr;
 public:
  //! Construit un tableau vide
  MDSpan() : MDSpan(nullptr,0){}
  //! Construit un tableau
  MDSpan(DataType* ptr,Int64 dim1_size)
  {
    m_dimensions[0] = dim1_size;
    m_ptr = ptr;
  }

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_dimensions[0]; }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    return i;
  }
  //! Valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType& operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    return m_ptr[i];
  }
  //! Pointeur sur la valeur pour l'élément \a i
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    return m_ptr+i;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 2D.
template<class DataType>
class MDSpan<DataType,2>
: public MDSpanBase<DataType,2>
{
  using BaseClass = MDSpanBase<DataType,2>;
  using BaseClass::m_dimensions;
  using BaseClass::m_ptr;
 public:
  //! Construit un tableau vide
  MDSpan() : MDSpan(nullptr,0,0){}
  //! Construit une vue
  MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size)
  {
    m_dimensions[0] = dim1_size;
    m_dimensions[1] = dim2_size;
    m_ptr = ptr;
  }

 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_dimensions[0]; }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int64 dim2Size() const { return m_dimensions[1]; }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    ARCCORE_CHECK_AT(j,m_dimensions[1]);
    return m_dimensions[1]*i + j;
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 3D.
template<class DataType>
class MDSpan<DataType,3>
: public MDSpanBase<DataType,3>
{
  using BaseClass = MDSpanBase<DataType,3>;
  using BaseClass::m_dimensions;
  using BaseClass::m_ptr;
 public:
  //! Construit un tableau vide
  MDSpan() : MDSpan(0,0,0){}
  //! Construit une vue
  MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    _setSize(ptr,dim1_size,dim2_size,dim3_size);
  }
 private:
  void _setSize(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    m_dim23_size = dim2_size * dim3_size;

    m_dimensions[0] = dim1_size;
    m_dimensions[1] = dim2_size;
    m_dimensions[2] = dim3_size;
    m_ptr = ptr;
  }
 public:
  //! Valeur de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_dimensions[0]; }
  //! Valeur de la deuxième dimension
  ARCCORE_HOST_DEVICE Int64 dim2Size() const { return m_dimensions[1]; }
  //! Valeur de la troisième dimension
  ARCCORE_HOST_DEVICE Int64 dim3Size() const { return m_dimensions[2]; }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    ARCCORE_CHECK_AT(j,m_dimensions[1]);
    ARCCORE_CHECK_AT(k,m_dimensions[2]);
    return (m_dim23_size*i) + m_dimensions[2]*j + k;
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
 private:
  Int64 m_dim23_size = 0; //!< dim2 * dim3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 4D.
template<class DataType>
class MDSpan<DataType,4>
: public MDSpanBase<DataType,4>
{
  using BaseClass = MDSpanBase<DataType,4>;
  using BaseClass::m_dimensions;
  using BaseClass::m_ptr;
 public:
  //! Construit un tableau vide
  MDSpan() : MDSpan(nullptr,0,0,0,0){}
  //! Construit une vue
  MDSpan(DataType* ptr,Int64 dim1_size,Int64 dim2_size,
         Int64 dim3_size,Int64 dim4_size)
  {
    setSize(ptr,dim1_size,dim2_size,dim3_size,dim4_size);
  }
 private:
  void _setSize(DataType* ptr,Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    m_dim34_size = dim3_size*dim4_size;
    m_dim234_size = m_dim34_size*dim2_size;
    m_dimensions[0] = dim1_size;
    m_dimensions[1] = dim2_size;
    m_dimensions[2] = dim3_size;
    m_dimensions[3] = dim4_size;
    m_ptr = ptr;
  }

 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return m_dimensions[0]; }
  //! Valeur de la deuxième dimension
  Int64 dim2Size() const { return m_dimensions[1]; }
  //! Valeur de la troisième dimension
  Int64 dim3Size() const { return m_dimensions[2]; }
  //! Valeur de la quatrième dimension
  Int64 dim4Size() const { return m_dimensions[3]; }
 public:
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    ARCCORE_CHECK_AT(i,m_dimensions[0]);
    ARCCORE_CHECK_AT(j,m_dimensions[1]);
    ARCCORE_CHECK_AT(k,m_dimensions[2]);
    ARCCORE_CHECK_AT(l,m_dimensions[3]);
    return (m_dim234_size*i) + m_dim34_size*j + m_dimensions[3]*k + l;
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
 private:
  Int64 m_dim34_size = 0; //!< dim3 * dim4
  Int64 m_dim234_size = 0; //!< dim2 * dim3 * dim4
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
  Int64 extent(int i) const { return m_extents.extent(i); }
 protected:
  NumArrayBase() : m_data(platform::getDefaultDataAllocator()){}
  void _resize()
  {
    Int64 full_size = extent(0);
    // TODO: vérifier débordement.
    for (int i=1; i<RankValue; ++i )
      full_size *= extent(i);
    m_total_nb_element = full_size;
    m_data.resize(full_size);
    m_ptr = m_data.data();
  }
 public:
  void fill(const DataType& v) { m_data.fill(v); }
  DataType* _internalData() { return m_ptr; }
  Int32 nbDimension() const { return RankValue; }
  Span<const Int64> extents() const { return m_extents.extents(); }
 protected:
  DataType* m_ptr = nullptr;
  ArrayExtentsWithOffset<RankValue> m_extents;
  UniqueArray<DataType> m_data;
  Int64 m_total_nb_element = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType,int rank>
class NumArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 1D.
template<class DataType>
class NumArray<DataType,1>
: public NumArrayBase<DataType,1>
{
  using BaseClass = NumArrayBase<DataType,1>;
  using BaseClass::m_ptr;
  using BaseClass::extent;
  using BaseClass::m_extents;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0){}
  //! Construit un tableau en le recopiant
  NumArray(const NumArray<DataType,1>& rhs) : BaseClass(rhs){}
  //! Construit un tableau
  NumArray(Int64 dim1_size)
  {
    resize(dim1_size);
  }
  void resize(Int64 dim1_size)
  {
    m_extents.setSize(dim1_size);
    this->_resize();
  }
 public:
  //! Valeur de la première dimension
  Int64 dim1Size() const { return this->extent(0); }
 public:
  //! Valeur pour l'élément \a i
  DataType operator()(Int64 i) const
  {
    return m_ptr[m_extents.offset(i)];
  }
  //! Positionne la valeur pour l'élément \a i
  DataType& s(Int64 i)
  {
    return m_ptr[m_extents.offset(i)];
  }
 public:
  operator MDSpan<DataType,1> () { return span(); }
  operator MDSpan<const DataType,1> () const { return constSpan(); }
 public:
  MDSpan<DataType,1> span()
  { return MDSpan<DataType,1>(m_ptr,extent(0)); }
  MDSpan<const DataType,1> span() const { return this->constSpan(); }
  MDSpan<const DataType,1> constSpan() const
  { return MDSpan<const DataType,1>(m_ptr,extent(0)); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 2D.
template<class DataType>
class NumArray<DataType,2>
: public NumArrayBase<DataType,2>
{
  using BaseClass = NumArrayBase<DataType,2>;
  using BaseClass::m_ptr;
  using BaseClass::extent;
  using BaseClass::m_extents;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0,0){}
  //! Construit un tableau en le recopiant
  NumArray(const NumArray<DataType,2>& rhs) : BaseClass(rhs){}
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size)
  {
    resize(dim1_size,dim2_size);
  }
  void resize(Int64 dim1_size,Int64 dim2_size)
  {
    m_extents.setSize(dim1_size,dim2_size);
    this->_resize();
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
    return m_ptr[m_extents.offset(i,j)];
  }
  //! Positionne la valeur pour l'élément \a i,j
  DataType& s(Int64 i,Int64 j)
  {
    return m_ptr[m_extents.offset(i,j)];
  }
 public:
  MDSpan<DataType,2> span()
  { return MDSpan<DataType,2>(m_ptr,extent(0),extent(1)); }
  MDSpan<const DataType,2> span() const { return this->constSpan(); }
  MDSpan<const DataType,2> constSpan() const
  { return MDSpan<const DataType,2>(m_ptr,extent(0),extent(1)); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 3D.
template<class DataType>
class NumArray<DataType,3>
: public NumArrayBase<DataType,3>
{
  using BaseClass = NumArrayBase<DataType,3>;
  using BaseClass::m_ptr;
  using BaseClass::extent;
  using BaseClass::m_extents;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0,0,0){}
  //! Construit un tableau en le recopiant
  NumArray(const NumArray<DataType,3>& rhs) : BaseClass(rhs){}
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    resize(dim1_size,dim2_size,dim3_size);
  }
  void resize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size);
    this->_resize();
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
    return m_ptr[m_extents.offset(i,j,k)];
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& s(Int64 i,Int64 j,Int64 k)
  {
    return m_ptr[m_extents.offset(i,j,k)];
  }
 public:
  MDSpan<DataType,3> span()
  { return MDSpan<DataType,3>(m_ptr,extent(0),extent(1),extent(2)); }
  MDSpan<const DataType,3> span() const { return this->constSpan(); }
  MDSpan<const DataType,3> constSpan() const
  { return MDSpan<const DataType,3>(m_ptr,extent(0),extent(1),extent(2)); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Tableaux 4D.
template<class DataType>
class NumArray<DataType,4>
: public NumArrayBase<DataType,4>
{
  using BaseClass = NumArrayBase<DataType,4>;
  using BaseClass::m_ptr;
  using BaseClass::extent;
  using BaseClass::m_extents;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0,0,0,0){}
  //! Construit une vue
  NumArray(Int64 dim1_size,Int64 dim2_size,
           Int64 dim3_size,Int64 dim4_size)
  {
    resize(dim1_size,dim2_size,dim3_size,dim4_size);
  }
  void resize(Int64 dim1_size,Int64 dim2_size,Int64 dim3_size,Int64 dim4_size)
  {
    m_extents.setSize(dim1_size,dim2_size,dim3_size,dim4_size);
    this->_resize();
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
    return m_ptr[m_extents.offset(i,j,k,l)];
  }
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& s(Int64 i,Int64 j,Int64 k,Integer l)
  {
    return m_ptr[m_extents.offset(i,j,k,l)];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
