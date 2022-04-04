﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExtents.h                                              (C) 2000-2022 */
/*                                                                           */
/* Gestion du nombre d'éléments par dimension pour les tableaux N-dimensions.*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYEXTENTS_H
#define ARCANE_UTILS_ARRAYEXTENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/ArrayBoundsIndex.h"
#include "arcane/utils/ArrayLayout.h"

#include "arccore/base/Span.h"

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
 * \brief Spécialisation de ArrayStrideBase pour les tableaux de dimension 0 (les scalaires)
 */
template<>
class ArrayStridesBase<0>
{
 public:
  ArrayStridesBase() = default;
  //! Valeur du pas de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return {}; }
  //! Value totale du pas
  ARCCORE_HOST_DEVICE Int64 totalStride() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayStridesBase<0> fromSpan([[maybe_unused]] Span<const Int32> strides)
  {
    // TODO: vérifier la taille de \a strides
    return {};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour conserver le pas dans chaque dimension.
 *
 * Le pas pour une dimension est la distance en mémoire entre deux éléments
 * du tableau pour cette dimension. En général le pas est égal au nombre
 * d'éléments dans la dimension sauf si on utilise des marges (padding) par
 * exemple pour aligner certaines dimensions.
 */
template<int RankValue>
class ArrayStridesBase
{
 public:
  ARCCORE_HOST_DEVICE ArrayStridesBase()
  {
    for( int i=0; i<RankValue; ++i )
      m_strides[i] = 0;
  }
  //! Valeur du pas de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE Int32 stride(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE Int32 operator()(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return { m_strides, RankValue }; }
  //! Valeur totale du pas
  ARCCORE_HOST_DEVICE Int64 totalStride() const
  {
    Int64 nb_element = 1;
    for (int i=0; i<RankValue; i++)
      nb_element *= m_strides[i];
    return nb_element;
  }
  // Instance contenant les dimensions après la première
  ARCCORE_HOST_DEVICE ArrayStridesBase<RankValue-1> removeFirstStride() const
  {
    return ArrayStridesBase<RankValue-1>::fromSpan({m_strides+1,RankValue-1});
  }
  /*!
   * \brief Construit une instance à partir des valeurs données dans \a stride.
   * \pre stride.size() == RankValue.
   */
  ARCCORE_HOST_DEVICE static ArrayStridesBase<RankValue> fromSpan(Span<const Int32> strides)
  {
    ArrayStridesBase<RankValue> v;
    // TODO: vérifier la taille
    for( int i=0; i<RankValue; ++i )
      v.m_strides[i] = strides[i];
    return v;
  }
 protected:
  Int32 m_strides[RankValue];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de ArrayExtentsBase pour les tableaux de dimension 0 (les scalaires)
 */
template<>
class ArrayExtentsBase<0>
{
 public:
  ArrayExtentsBase() = default;
  //! Nombre d'élément de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return {}; }
  //! Nombre total d'eléments
  ARCCORE_HOST_DEVICE Int32 totalNbElement() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayExtentsBase<0> fromSpan([[maybe_unused]] Span<const Int32> extents)
  {
    // TODO: vérifier la taille de \a extents
    return {};
  }
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
  ARCCORE_HOST_DEVICE constexpr ArrayExtentsBase()
  {
    for( int i=0; i<RankValue; ++i )
      m_extents[i] = 0;
  }
 protected:
  explicit ARCCORE_HOST_DEVICE ArrayExtentsBase(SmallSpan<const Int32> extents)
  {
    Integer n = extents.size();
    Integer vn = math::min(n,RankValue);
    for( int i=0; i<vn; ++i )
      m_extents[i] = extents[i];
    for( int i=vn; i<RankValue; ++i )
      m_extents[i] = 0;
  }
 public:
  //! Nombre d'élément de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE Int32 extent(int i) const { return m_extents[i]; }
  //! Positionne à \a v le nombre d'éléments de la i-ème dimension
  ARCCORE_HOST_DEVICE void setExtent(int i,Int32 v) { m_extents[i] = v; }
  ARCCORE_HOST_DEVICE Int32 operator()(int i) const { return m_extents[i]; }
  ARCCORE_HOST_DEVICE SmallSpan<const Int32> asSpan() const { return { m_extents.data(), RankValue }; }
  ARCCORE_HOST_DEVICE std::array<Int32,RankValue> asStdArray() const { return m_extents; }
  //! Nombre total d'eléments
  ARCCORE_HOST_DEVICE constexpr Int64 totalNbElement() const
  {
    Int64 nb_element = 1;
    for (int i=0; i<RankValue; i++)
      nb_element *= m_extents[i];
    return nb_element;
  }
  // Instance contenant les dimensions après la première
  ARCCORE_HOST_DEVICE ArrayExtentsBase<RankValue-1> removeFirstExtent() const
  {
    return ArrayExtentsBase<RankValue-1>::fromSpan({m_extents.data()+1,RankValue-1});
  }
  /*!
   * \brief Construit une instance à partir des valeurs données dans \a extents.
   */
  ARCCORE_HOST_DEVICE static ArrayExtentsBase<RankValue> fromSpan(SmallSpan<const Int32> extents)
  {
    return ArrayExtentsBase<RankValue>(extents);
  }
 protected:
  std::array<Int32,RankValue> m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtents<1>
: public ArrayExtentsBase<1>
{
 public:

  using BaseClass = ArrayExtentsBase<1>;

 public:

  ArrayExtents() = default;
  ArrayExtents(BaseClass rhs) : BaseClass(rhs){}
  ARCCORE_HOST_DEVICE explicit ArrayExtents(Int32 dim1_size)
  {
    setSize(dim1_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size)
  {
    m_extents[0] = dim1_size;
  }

 protected:

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] ArrayBoundsIndex<1> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(),m_extents[0]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtents<2>
: public ArrayExtentsBase<2>
{
 public:

  using BaseClass = ArrayExtentsBase<2>;

 public:

  ArrayExtents() = default;
  ArrayExtents(BaseClass rhs) : BaseClass(rhs){}
  ARCCORE_HOST_DEVICE ArrayExtents(Int32 dim1_size,Int32 dim2_size)
  {
    setSize(dim1_size,dim2_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size,Int32 dim2_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
  }

 protected:

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] ArrayBoundsIndex<2> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(),m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1(),m_extents[1]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtents<3>
: public ArrayExtentsBase<3>
{
 public:

  using BaseClass = ArrayExtentsBase<3>;

 public:

  ArrayExtents() = default;
  ArrayExtents(BaseClass rhs) : BaseClass(rhs){}
  ARCCORE_HOST_DEVICE ArrayExtents(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    setSize(dim1_size,dim2_size,dim3_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
  }

 protected:

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] ArrayBoundsIndex<3> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(),m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1(),m_extents[1]);
    ARCCORE_CHECK_AT(idx.id2(),m_extents[2]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayExtents<4>
: public ArrayExtentsBase<4>
{
 public:
  using BaseClass = ArrayExtentsBase<4>;
 public:
  ArrayExtents() = default;
  ArrayExtents(BaseClass rhs) : BaseClass(rhs){}
  ARCCORE_HOST_DEVICE ArrayExtents(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,Int32 dim4_size)
  {
    setSize(dim1_size,dim2_size,dim3_size,dim4_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,Int32 dim4_size)
  {
    m_extents[0] = dim1_size;
    m_extents[1] = dim2_size;
    m_extents[2] = dim3_size;
    m_extents[3] = dim4_size;
  }

 protected:

  ARCCORE_HOST_DEVICE void _checkIndex([[maybe_unused]] ArrayBoundsIndex<4> idx) const
  {
    ARCCORE_CHECK_AT(idx.id0(),m_extents[0]);
    ARCCORE_CHECK_AT(idx.id1(),m_extents[1]);
    ARCCORE_CHECK_AT(idx.id2(),m_extents[2]);
    ARCCORE_CHECK_AT(idx.id3(),m_extents[3]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
class ArrayExtentsWithOffset<1,LayoutType>
: private ArrayExtents<1>
{
 public:
  using BaseClass = ArrayExtents<1>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::asSpan;
  using BaseClass::asStdArray;
  using BaseClass::totalNbElement;
  using Layout = LayoutType;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<1> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i) const
  {
    BaseClass::_checkIndex(i);
    return i;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<1> idx) const
  {
    BaseClass::_checkIndex(idx.id0());
    return idx.id0();
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size)
  {
    BaseClass::setSize(dim1_size);
  }
  ARCCORE_HOST_DEVICE void setSize(ArrayExtents<1> extents)
  {
    BaseClass::setSize(extents(0));
  }
  BaseClass extents() const { const BaseClass* b = this; return *b; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
class ArrayExtentsWithOffset<2,LayoutType>
: private ArrayExtents<2>
{
 public:
  using BaseClass = ArrayExtents<2>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::asSpan;
  using BaseClass::asStdArray;
  using BaseClass::totalNbElement;
  using Layout = LayoutType;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<2> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j) const
  {
    return offset({i,j});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<2> idx) const
  {
    BaseClass::_checkIndex(idx);
    return Layout::offset(idx,m_extents[Layout::LastExtent]);
  }
  ARCCORE_HOST_DEVICE void setSize(Int32 dim1_size,Int32 dim2_size)
  {
    BaseClass::setSize(dim1_size,dim2_size);
  }
  ARCCORE_HOST_DEVICE void setSize(ArrayExtents<2> dims)
  {
    this->setSize(dims(0),dims(1));
  }
  BaseClass extents() const { const BaseClass* b = this; return *b; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
class ArrayExtentsWithOffset<3,LayoutType>
: private ArrayExtents<3>
{
 public:
  using BaseClass = ArrayExtents<3>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::asSpan;
  using BaseClass::asStdArray;
  using BaseClass::totalNbElement;
  using Layout = LayoutType;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<3> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j,Int32 k) const
  {
    return offset({i,j,k});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<3> idx) const
  {
    this->_checkIndex(idx);
    return Layout::offset(idx,m_extents[Layout::LastExtent],m_dim23_size);
  }
  void setSize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    BaseClass::setSize(dim1_size,dim2_size,dim3_size);
    _computeOffsets();
  }
  void setSize(ArrayExtents<3> dims)
  {
    this->setSize(dims(0),dims(1),dims(2));
  }
  BaseClass extents() const { const BaseClass* b = this; return *b; }
 protected:
  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    m_dim23_size = Layout::computeOffsetIndexes(m_extents);
  }
 private:
  Int64 m_dim23_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LayoutType>
class ArrayExtentsWithOffset<4,LayoutType>
: private ArrayExtents<4>
{
 public:
  using BaseClass = ArrayExtents<4>;
  using BaseClass::extent;
  using BaseClass::operator();
  using BaseClass::asSpan;
  using BaseClass::asStdArray;
  using BaseClass::totalNbElement;
  using Layout = LayoutType;
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<4> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return offset({i,j,k,l});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<4> idx) const
  {
    this->_checkIndex(idx);
    return (m_dim234_size*idx.largeId0()) + m_dim34_size*idx.largeId1() + m_extents[3]*idx.largeId2() + idx.largeId3();
  }
  void setSize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,Int32 dim4_size)
  {
    BaseClass::setSize(dim1_size,dim2_size,dim3_size,dim4_size);
    _computeOffsets();
  }
  void setSize(ArrayExtents<4> dims)
  {
    this->setSize(dims(0),dims(1),dims(2),dims(3));
  }
  BaseClass extents() const { const BaseClass* b = this; return *b; }
 protected:
  ARCCORE_HOST_DEVICE void _computeOffsets()
  {
    m_dim34_size = Int64(m_extents[2]) * Int64(m_extents[3]);
    m_dim234_size = Int64(m_dim34_size) * Int64(m_extents[1]);
  }
 private:
  Int64 m_dim34_size = 0; //!< dim3 * dim4
  Int64 m_dim234_size = 0; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
