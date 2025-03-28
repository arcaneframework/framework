// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FixedArray.h                                                (C) 2000-2025 */
/*                                                                           */
/* Tableau 1D de taille fixe.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FIXEDARRAY_H
#define ARCANE_UTILS_FIXEDARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Tableau 1D de taille fixe.
 *
 * Cette classe est similaire à std::array avec les différences suivantes:
 *
 * - le nombre d'élément est un 'Int32'.
 * - les éléments sont initialisés avec le constructeur par défaut
 * - en mode 'Check', vérifie les débordements de tableau
 *
 * Cette classe propose aussi des conversions vers ArrayView, ConstArrayView
 * et SmallSpan.
 */
template <typename T, Int32 NbElement>
class FixedArray final
{
  static_assert(NbElement >= 0, "NbElement has to positive");

 public:

  using value_type = T;
  using size_type = Int32;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  using iterator = typename std::array<T, NbElement>::iterator;
  using const_iterator = typename std::array<T, NbElement>::const_iterator;

 public:

  //! Créé un tableau en initialisant les éléments avec le constructeur par défaut de \a T
  constexpr FixedArray()
  : m_value({})
  {}
  //! Créé un tableau en initialisant les éléments de \a x
  constexpr FixedArray(std::array<T, NbElement> x)
  : m_value(std::move(x))
  {}
  //! Recopie \a x dans l'instance
  constexpr FixedArray<T,NbElement>& operator=(std::array<T, NbElement> x)
  {
    m_value = std::move(x);
    return *this;
  }

 public:

  //! Valeur du \a i-ème élément
  constexpr ARCCORE_HOST_DEVICE T& operator[](Int32 index)
  {
    ARCANE_CHECK_AT(index, NbElement);
    return m_value[index];
  }
  //! Valeur du \a i-ème élément
  constexpr ARCCORE_HOST_DEVICE const T& operator[](Int32 index) const
  {
    ARCANE_CHECK_AT(index, NbElement);
    return m_value[index];
  }
  //! Vue modifiable sur le tableau
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, NbElement> span() { return { m_value.data(), NbElement }; }
  //! Vue non modifiable sur le tableau
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const T, NbElement> span() const { return { m_value.data(), NbElement }; }
  //! Vue modifiable sur le tableau
  constexpr ARCCORE_HOST_DEVICE ArrayView<T> view() { return { NbElement, m_value.data() }; }
  //! Vue non modifiable sur le tableau
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<T> view() const { return { NbElement, m_value.data() }; }
  constexpr ARCCORE_HOST_DEVICE const T* data() const { return m_value.data(); }
  constexpr ARCCORE_HOST_DEVICE T* data() { return m_value.data(); }

  //! Nombre d'éléments tu tableau
  static constexpr Int32 size() { return NbElement; }

 public:

  //! Itérateur sur le début du tableau
  constexpr iterator begin() { return m_value.begin(); }
  //! Itérateur sur la fin du tableau
  constexpr iterator end() { return m_value.end(); }
  //! Itérateur constant sur le début du tableau
  constexpr const_iterator begin() const { return m_value.begin(); }
  //! Itérateur constant la fin du tableau
  constexpr const_iterator end() const { return m_value.end(); }

 private:

  std::array<T, NbElement> m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
