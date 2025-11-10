// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDSpan.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Vue sur un tableaux multi-dimensionnel pour les types numériques.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MDSPAN_H
#define ARCCORE_BASE_MDSPAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayExtents.h"
#include "arccore/base/ArrayBounds.h"
#include "arccore/base/NumericTraits.h"
#include "arccore/base/ArrayLayout.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues multi-dimensionnelles.
 *
 * Cette classe s'inspire la classe std::mdspan en cours de définition
 * (voir http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0009r12.html)
 *
 * Cette classe est utilisée pour gérer les vues sur les tableaux tels que
 * NumArray. Les méthodes de cette classe sont accessibles sur accélérateur.
 *
 * Pour plus d'informations, se reporter à la page \ref arcanedoc_core_types_numarray.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpan
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArray<UnqualifiedValueType, Extents, LayoutPolicy>;
  // Pour que MDSpan<const T> ait accès à MDSpan<T>
  friend class MDSpan<const UnqualifiedValueType, Extents, LayoutPolicy>;
  using ThatClass = MDSpan<DataType, Extents, LayoutPolicy>;
  static constexpr bool IsConst = std::is_const_v<DataType>;

 public:

  using value_type = DataType;
  using ExtentsType = Extents;
  using LayoutPolicyType = LayoutPolicy;
  using MDIndexType = typename Extents::MDIndexType;
  using LoopIndexType = MDIndexType;
  using ArrayExtentsWithOffsetType = ArrayExtentsWithOffset<Extents, LayoutPolicy>;
  using DynamicDimsType = typename Extents::DynamicDimsType;
  using RemovedFirstExtentsType = typename Extents::RemovedFirstExtentsType;

  // Pour compatibilité. A supprimer pour cohérence avec les autres 'using'
  using ArrayBoundsIndexType = typename Extents::MDIndexType;
  using IndexType = typename Extents::MDIndexType;

 public:

  MDSpan() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpan(DataType* ptr, ArrayExtentsWithOffsetType extents)
  : m_ptr(ptr)
  , m_extents(extents)
  {
  }
  constexpr ARCCORE_HOST_DEVICE MDSpan(DataType* ptr, const DynamicDimsType& dims)
  : m_ptr(ptr)
  , m_extents(dims)
  {}
  // Constructeur MDSpan<const T> à partir d'un MDSpan<T>
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpan(const MDSpan<X, Extents>& rhs)
  : m_ptr(rhs.m_ptr)
  , m_extents(rhs.m_extents)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpan(SmallSpan<DataType> v) requires(Extents::isDynamic1D() && !IsConst)
  : m_ptr(v.data())
  , m_extents(DynamicDimsType(v.size()))
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpan(SmallSpan<const DataType> v) requires(Extents::isDynamic1D() && IsConst)
  : m_ptr(v.data())
  , m_extents(DynamicDimsType(v.size()))
  {}
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(SmallSpan<DataType> v) requires(Extents::isDynamic1D() && !IsConst)
  {
    m_ptr = v.data();
    m_extents = DynamicDimsType(v.size());
    return (*this);
  }
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(SmallSpan<const DataType> v) requires(Extents::isDynamic1D() && IsConst)
  {
    m_ptr = v.data();
    m_extents = DynamicDimsType(v.size());
    return (*this);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE DataType* _internalData() { return m_ptr; }
  constexpr ARCCORE_HOST_DEVICE const DataType* _internalData() const { return m_ptr; }

 public:

  ArrayExtents<Extents> extents() const
  {
    return m_extents.extents();
  }
  ArrayExtentsWithOffsetType extentsWithOffset() const
  {
    return m_extents;
  }

 public:

  //! Valeur de la première dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const requires(Extents::rank() >= 1) { return m_extents.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const requires(Extents::rank() >= 2) { return m_extents.extent1(); }
  //! Valeur de la troisième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent2() const requires(Extents::rank() >= 3) { return m_extents.extent2(); }
  //! Valeur de la quatrième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent3() const requires(Extents::rank() >= 4) { return m_extents.extent3(); }

 public:

  //! Valeur pour l'élément \a i,j,k,l
  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j, Int32 k, Int32 l) const requires(Extents::rank() == 4)
  {
    return m_extents.offset(i, j, k, l);
  }
  //! Valeur pour l'élément \a i,j,k
  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j, Int32 k) const requires(Extents::rank() == 3)
  {
    return m_extents.offset(i, j, k);
  }
  //! Valeur pour l'élément \a i,j
  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j) const requires(Extents::rank() == 2)
  {
    return m_extents.offset(i, j);
  }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i) const requires(Extents::rank() == 1) { return m_extents.offset(i); }

  //! Valeur pour l'élément \a idx
  constexpr ARCCORE_HOST_DEVICE Int64 offset(MDIndexType idx) const
  {
    return m_extents.offset(idx);
  }

 public:

  //! Valeur pour l'élément \a i,j,k,l
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j, Int32 k, Int32 l) const requires(Extents::rank() == 4)
  {
    return m_ptr[offset(i, j, k, l)];
  }
  //! Valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j, Int32 k) const requires(Extents::rank() == 3)
  {
    return m_ptr[offset(i, j, k)];
  }
  //! Valeur pour l'élément \a i,j
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j) const requires(Extents::rank() == 2)
  {
    return m_ptr[offset(i, j)];
  }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i) const requires(Extents::rank() == 1) { return m_ptr[offset(i)]; }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType operator[](Int32 i) const requires(Extents::rank() == 1) { return m_ptr[offset(i)]; }

  //! Valeur pour l'élément \a idx
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(MDIndexType idx) const
  {
    return m_ptr[offset(idx)];
  }

 public:

  //! Pointeur sur la valeur pour l'élément \a i,j,k
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j, Int32 k, Int32 l) const requires(Extents::rank() == 4)
  {
    return m_ptr + offset(i, j, k, l);
  }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j, Int32 k) const requires(Extents::rank() == 3)
  {
    return m_ptr + offset(i, j, k);
  }
  //! Pointeur sur la valeur pour l'élément \a i,j
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j) const requires(Extents::rank() == 2)
  {
    return m_ptr + offset(i, j);
  }
  //! Pointeur sur la valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i) const requires(Extents::rank() == 1) { return m_ptr + offset(i); }

  //! Pointeur sur la valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(MDIndexType idx) const
  {
    return m_ptr + offset(idx);
  }

 public:

  /*!
   * \brief Retourne une vue de dimension (N-1) à partir de l'élément d'indice \a i.
   *
   * Par exemple:
   * \code
   *   MDSpan<Real, MDDim3> span3 = ...;
   *   MDSpan<Real, MDDim2> sliced_span = span3.slice(5);
   *   // sliced_span(i,i) <=> span3(5,i,j);
   * \endcode
   *
   * \warning Cela n'est valide que si \a LayoutPolicy est \a RightLayout.
   */
  ARCCORE_HOST_DEVICE MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>
  slice(Int32 i) const requires(Extents::rank() >= 2 && std::is_base_of_v<RightLayout, LayoutPolicy>)
  {
    auto new_extents = m_extents.extents().removeFirstExtent().dynamicExtents();
    std::array<Int32, ExtentsType::rank()> indexes = {};
    indexes[0] = i;
    DataType* base_ptr = this->ptrAt(MDIndexType(indexes));
    return MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>(base_ptr, new_extents);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE MDSpan<const DataType, Extents, LayoutPolicy> constSpan() const
  {
    return MDSpan<const DataType, Extents, LayoutPolicy>(m_ptr, m_extents);
  }

  constexpr ARCCORE_HOST_DEVICE MDSpan<const DataType, Extents, LayoutPolicy> constMDSpan() const
  {
    return MDSpan<const DataType, Extents, LayoutPolicy>(m_ptr, m_extents);
  }

  constexpr ARCCORE_HOST_DEVICE Span<DataType> to1DSpan() const
  {
    return { m_ptr, m_extents.totalNbElement() };
  }

  constexpr SmallSpan<DataType> to1DSmallSpan() requires(Extents::rank() == 1)
  {
    return { _internalData(), extent0() };
  }
  constexpr SmallSpan<const DataType> to1DSmallSpan() const requires(Extents::rank() == 1)
  {
    return to1DConstSmallSpan();
  }
  constexpr SmallSpan<const DataType> to1DConstSmallSpan() const requires(Extents::rank() == 1)
  {
    return { _internalData(), extent0() };
  }

 private:

  DataType* m_ptr = nullptr;
  ArrayExtentsWithOffsetType m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
