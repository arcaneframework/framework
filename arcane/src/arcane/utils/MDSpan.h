// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDSpan.h                                                    (C) 2000-2023 */
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

//! Spécialisation intérmédiaire
template <typename DataType, int Rank, typename Extents, typename LayoutPolicy>
class MDSpanIntermediate;

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
 * Cette classe ne doit pas être utilisée directement. Il faut utiliser MDSpan
 * à la place.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpanBase
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType, Extents, LayoutPolicy>;
  // Pour que MDSpan<const T> ait accès à MDSpan<T>
  friend class MDSpanBase<const UnqualifiedValueType, Extents, LayoutPolicy>;

 public:

  using ExtentsType = Extents;
  using IndexType = typename Extents::IndexType;
  using ArrayExtentsWithOffsetType = ArrayExtentsWithOffset<Extents, LayoutPolicy>;
  using DynamicDimsType = typename Extents::DynamicDimsType;
  // Pour compatibilité. A supprimer pour cohérence avec les autres 'using'
  using ArrayBoundsIndexType = typename Extents::IndexType;

 public:

  MDSpanBase() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpanBase(DataType* ptr, ArrayExtentsWithOffsetType extents)
  : m_ptr(ptr)
  , m_extents(extents)
  {
  }
  constexpr ARCCORE_HOST_DEVICE MDSpanBase(DataType* ptr, const DynamicDimsType& dims)
  : m_ptr(ptr)
  , m_extents(dims)
  {}
  // Constructeur MDSpan<const T> à partir d'un MDSpan<T>
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpanBase(const MDSpanBase<X, Extents>& rhs)
  : m_ptr(rhs.m_ptr)
  , m_extents(rhs.m_extents)
  {}

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

  constexpr ARCCORE_HOST_DEVICE Int64 offset(IndexType idx) const
  {
    return m_extents.offset(idx);
  }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(IndexType idx) const
  {
    return m_ptr[offset(idx)];
  }
  //! Pointeur sur la valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(IndexType idx) const
  {
    return m_ptr + offset(idx);
  }

 public:

  constexpr ARCCORE_HOST_DEVICE MDSpanBase<const DataType, Extents> constSpan() const
  {
    return MDSpanBase<const DataType, Extents>(m_ptr, m_extents);
  }
  constexpr ARCCORE_HOST_DEVICE Span<DataType> to1DSpan() const
  {
    return { m_ptr, m_extents.totalNbElement() };
  }

 protected:

  DataType* m_ptr = nullptr;
  ArrayExtentsWithOffsetType m_extents;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation d'une vue multi-dimensionnelle à 1 dimension.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpanIntermediate<DataType, 1, Extents, LayoutPolicy>
: public MDSpanBase<DataType, Extents, LayoutPolicy>
{
 protected:

  using ExtentsType = Extents;
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType, Extents, LayoutPolicy>;
  using BaseClass = MDSpanBase<DataType, Extents, LayoutPolicy>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;

 public:

  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
  using ArrayExtentsWithOffsetType = typename BaseClass::ArrayExtentsWithOffsetType;
  using DynamicDimsType = typename Extents::DynamicDimsType;

 public:

  //! Construit une vue vide
  MDSpanIntermediate() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, ArrayExtentsWithOffsetType extents_and_offset)
  : BaseClass(ptr, extents_and_offset)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, const DynamicDimsType& dims)
  : BaseClass(ptr, dims)
  {
  }
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(const MDSpan<X, ExtentsType>& rhs)
  : BaseClass(rhs)
  {}

 public:

  //! Valeur de la première dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extents.extent0(); }

 public:

  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i) const { return m_extents.offset(i); }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i) const { return m_ptr[offset(i)]; }
  //! Pointeur sur la valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i) const { return m_ptr + offset(i); }
  //! Valeur pour l'élément \a i
  constexpr ARCCORE_HOST_DEVICE DataType operator[](Int32 i) const { return m_ptr[offset(i)]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation d'une vue multi-dimensionnelle à 2 dimensions.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpanIntermediate<DataType, 2, Extents, LayoutPolicy>
: public MDSpanBase<DataType, Extents, LayoutPolicy>
{
 protected:

  using ExtentsType = Extents;
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType, Extents, LayoutPolicy>;
  using BaseClass = MDSpanBase<DataType, Extents, LayoutPolicy>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;

 public:

  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
  using ArrayExtentsWithOffsetType = typename BaseClass::ArrayExtentsWithOffsetType;
  using DynamicDimsType = typename Extents::DynamicDimsType;

 protected:

  //! Construit un tableau vide
  MDSpanIntermediate() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, ArrayExtentsWithOffsetType extents_and_offset)
  : BaseClass(ptr, extents_and_offset)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, const DynamicDimsType& dims)
  : BaseClass(ptr, dims)
  {
  }
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(const MDSpan<X, ExtentsType>& rhs)
  : BaseClass(rhs)
  {}

 public:

  //! Valeur de la première dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extents.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extents.extent1(); }

 public:

  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j) const { return m_extents.offset(i, j); }
  //! Valeur pour l'élément \a i,j
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j) const { return m_ptr[offset(i, j)]; }
  //! Pointeur sur la valeur pour l'élément \a i,j
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j) const { return m_ptr + offset(i, j); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation d'une vue multi-dimensionnelle à 3 dimensions.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpanIntermediate<DataType, 3, Extents, LayoutPolicy>
: public MDSpanBase<DataType, Extents, LayoutPolicy>
{
 protected:

  using ExtentsType = Extents;
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType, Extents, LayoutPolicy>;
  using BaseClass = MDSpanBase<DataType, Extents, LayoutPolicy>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
  using value_type = typename std::remove_cv<DataType>::type;

 public:

  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
  using ArrayExtentsWithOffsetType = typename BaseClass::ArrayExtentsWithOffsetType;
  using DynamicDimsType = typename Extents::DynamicDimsType;

 protected:

  //! Construit une vue vide
  MDSpanIntermediate() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, ArrayExtentsWithOffsetType extents_and_offset)
  : BaseClass(ptr, extents_and_offset)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, const DynamicDimsType& dims)
  : BaseClass(ptr, dims)
  {
  }
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(const MDSpan<X, ExtentsType>& rhs)
  : BaseClass(rhs)
  {}

 public:

  //! Valeur de la première dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extents.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extents.extent1(); }
  //! Valeur de la troisième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent2() const { return m_extents.extent2(); }

 public:

  ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j, Int32 k) const { return m_extents.offset(i, j, k); }
  //! Valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j, Int32 k) const { return m_ptr[offset(i, j, k)]; }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j, Int32 k) const { return m_ptr + offset(i, j, k); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue multi-dimensionnelle à 4 dimensions.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class MDSpanIntermediate<DataType, 4, Extents, LayoutPolicy>
: public MDSpanBase<DataType, Extents, LayoutPolicy>
{
 protected:

  using BaseClass = MDSpanBase<DataType, Extents, LayoutPolicy>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;
  using UnqualifiedValueType = std::remove_cv_t<DataType>;

 public:

  using ExtentsType = Extents;
  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
  using ArrayExtentsWithOffsetType = typename BaseClass::ArrayExtentsWithOffsetType;
  using DynamicDimsType = typename Extents::DynamicDimsType;

 protected:

  //! Construit une vue vide
  MDSpanIntermediate() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, ArrayExtentsWithOffsetType extents_and_offset)
  : BaseClass(ptr, extents_and_offset)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(DataType* ptr, const DynamicDimsType& dims)
  : BaseClass(ptr, dims)
  {
  }
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpanIntermediate(const MDSpan<X, Extents>& rhs)
  : BaseClass(rhs)
  {}

 public:

  //! Valeur de la première dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent0() const { return m_extents.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent1() const { return m_extents.extent1(); }
  //! Valeur de la troisième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent2() const { return m_extents.extent2(); }
  //! Valeur de la quatrième dimension
  constexpr ARCCORE_HOST_DEVICE Int32 extent3() const { return m_extents.extent3(); }

 public:

  constexpr ARCCORE_HOST_DEVICE Int64 offset(Int32 i, Int32 j, Int32 k, Int32 l) const
  {
    return m_extents.offset(i, j, k, l);
  }

 public:

  //! Valeur pour l'élément \a i,j,k,l
  constexpr ARCCORE_HOST_DEVICE DataType& operator()(Int32 i, Int32 j, Int32 k, Int32 l) const
  {
    return m_ptr[offset(i, j, k, l)];
  }
  //! Pointeur sur la valeur pour l'élément \a i,j,k
  constexpr ARCCORE_HOST_DEVICE DataType* ptrAt(Int32 i, Int32 j, Int32 k, Int32 l) const
  {
    return m_ptr + offset(i, j, k, l);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vues sur des tableaux multi-dimensionnels.
 *
 * \warning API en cours de définition.
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
: public MDSpanIntermediate<DataType, Extents::rank(), Extents, LayoutPolicy>
{
  using UnqualifiedValueType = std::remove_cv_t<DataType>;
  friend class NumArrayBase<UnqualifiedValueType, Extents, LayoutPolicy>;
  using BaseClass = MDSpanIntermediate<DataType, Extents::rank(), Extents, LayoutPolicy>;
  using BaseClass::m_extents;
  using BaseClass::m_ptr;

 public:

  using BaseClass::offset;
  using BaseClass::ptrAt;
  using BaseClass::operator();
  using ArrayExtentsWithOffsetType = typename BaseClass::ArrayExtentsWithOffsetType;
  using DynamicDimsType = typename Extents::DynamicDimsType;
  using ExtentsType = Extents;
  using RemovedFirstExtentsType = typename Extents::RemovedFirstExtentsType;
  using IndexType = typename BaseClass::IndexType;

 public:

  //! Construit une vue vide
  MDSpan() = default;
  constexpr ARCCORE_HOST_DEVICE MDSpan(DataType* ptr, ArrayExtentsWithOffsetType extents_and_offset)
  : BaseClass(ptr, extents_and_offset)
  {}
  constexpr ARCCORE_HOST_DEVICE MDSpan(DataType* ptr, const DynamicDimsType& dims)
  : BaseClass(ptr, dims)
  {
  }
  template <typename X = ExtentsType, typename = std::enable_if_t<X::nb_dynamic == 0, void>>
  explicit constexpr ARCCORE_HOST_DEVICE MDSpan(DataType* ptr)
  : BaseClass(ptr, DynamicDimsType{})
  {
  }
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, UnqualifiedValueType>>>
  constexpr ARCCORE_HOST_DEVICE MDSpan(const MDSpan<X, ExtentsType>& rhs)
  : BaseClass(rhs)
  {}

 public:

  ARCCORE_HOST_DEVICE MDSpan<const DataType, Extents, LayoutPolicy> constSpan() const
  {
    return MDSpan<const DataType, Extents, LayoutPolicy>(m_ptr, m_extents);
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
   */
  template <typename X = ExtentsType, typename = std::enable_if_t<X::rank() >= 2, void>>
  ARCCORE_HOST_DEVICE MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>
  slice(Int32 i) const
  {
    auto new_extents = m_extents.extents().removeFirstExtent().dynamicExtents();
    std::array<Int32, ExtentsType::rank()> indexes = {};
    indexes[0] = i;
    DataType* base_ptr = this->ptrAt(IndexType(indexes));
    return MDSpan<DataType, RemovedFirstExtentsType, LayoutPolicy>(base_ptr, new_extents);
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
