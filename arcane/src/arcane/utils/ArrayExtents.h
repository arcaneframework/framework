// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayExtents.h                                              (C) 2000-2021 */
/*                                                                           */
/* Gestion du nombre d'éléments par dimension pour les tableaux N-dimensions.*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYEXTENTS_H
#define ARCANE_UTILS_ARRAYEXTENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/ArrayBoundsIndex.h"

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
  ARCCORE_HOST_DEVICE SmallSpan<const Int64> asSpan() const { return {}; }
  //! Value totale du pas
  ARCCORE_HOST_DEVICE Int64 totalStride() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayStridesBase<0> fromSpan([[maybe_unused]] Span<const Int64> strides)
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
  ARCCORE_HOST_DEVICE Int64 stride(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE Int64 operator()(int i) const { return m_strides[i]; }
  ARCCORE_HOST_DEVICE SmallSpan<const Int64> asSpan() const { return { m_strides, RankValue }; }
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
  ARCCORE_HOST_DEVICE static ArrayStridesBase<RankValue> fromSpan(Span<const Int64> strides)
  {
    ArrayStridesBase<RankValue> v;
    // TODO: vérifier la taille
    for( int i=0; i<RankValue; ++i )
      v.m_strides[i] = strides[i];
    return v;
  }
 protected:
  Int64 m_strides[RankValue];
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
  ARCCORE_HOST_DEVICE SmallSpan<const Int64> asSpan() const { return {}; }
  //! Nombre total d'eléments
  ARCCORE_HOST_DEVICE Int64 totalNbElement() const { return 1; }
  ARCCORE_HOST_DEVICE static ArrayExtentsBase<0> fromSpan([[maybe_unused]] Span<const Int64> extents)
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
  //! Nombre d'élément de la \a i-ème dimension.
  ARCCORE_HOST_DEVICE Int64 extent(int i) const { return m_extents[i]; }
  //! Positionne à \a v le nombre d'éléments de la i-ème dimension
  ARCCORE_HOST_DEVICE void setExtent(int i,Int64 v) { m_extents[i] = v; }
  ARCCORE_HOST_DEVICE Int64 operator()(int i) const { return m_extents[i]; }
  ARCCORE_HOST_DEVICE SmallSpan<const Int64> asSpan() const { return { m_extents.data(), RankValue }; }
  ARCCORE_HOST_DEVICE std::array<Int64,RankValue> asStdArray() const { return m_extents; }
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
   * \pre extents.size() == RankValue.
   */
  ARCCORE_HOST_DEVICE static ArrayExtentsBase<RankValue> fromSpan(Span<const Int64> extents)
  {
    ArrayExtentsBase<RankValue> v;
    // TODO: vérifier la taille
    for( int i=0; i<RankValue; ++i )
      v.m_extents[i] = extents[i];
    return v;
  }
 protected:
  std::array<Int64,RankValue> m_extents;
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
  ARCCORE_HOST_DEVICE explicit ArrayExtents(Int64 dim1_size)
  {
    setSize(dim1_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size)
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
  ARCCORE_HOST_DEVICE ArrayExtents(Int64 dim1_size,Int64 dim2_size)
  {
    setSize(dim1_size,dim2_size);
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size)
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
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<1> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i) const
  {
    BaseClass::_checkIndex(i);
    return i;
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<1> idx) const
  {
    BaseClass::_checkIndex(idx.id0());
    return idx.id0();
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size)
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
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<2> rhs)
  : BaseClass(rhs)
  {
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j) const
  {
    return offset({i,j});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<2> idx) const
  {
    BaseClass::_checkIndex(idx);
    return m_extents[1]*idx.id0() + idx.id1();
  }
  ARCCORE_HOST_DEVICE void setSize(Int64 dim1_size,Int64 dim2_size)
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
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<3> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k) const
  {
    return offset({i,j,k});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<3> idx) const
  {
    this->_checkIndex(idx);
    return (m_dim23_size*idx.id0()) + m_extents[2]*idx.id1() + idx.id2();
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
  BaseClass extents() const { const BaseClass* b = this; return *b; }
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
 public:
  ArrayExtentsWithOffset() = default;
  ARCCORE_HOST_DEVICE ArrayExtentsWithOffset(ArrayExtents<4> rhs)
  : BaseClass(rhs)
  {
    _computeOffsets();
  }
  ARCCORE_HOST_DEVICE Int64 offset(Int64 i,Int64 j,Int64 k,Int64 l) const
  {
    return offset({i,j,k,l});
  }
  ARCCORE_HOST_DEVICE Int64 offset(ArrayBoundsIndex<4> idx) const
  {
    this->_checkIndex(idx);
    return (m_dim234_size*idx.id0()) + m_dim34_size*idx.id1() + m_extents[3]*idx.id2() + idx.id3();
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
  BaseClass extents() const { const BaseClass* b = this; return *b; }
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
