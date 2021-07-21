// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Span2.h                                                     (C) 2000-2018 */
/*                                                                           */
/* Vue d'un tableau 2D dont les dimensions utilisent des Int64.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_SPAN2_H
#define ARCCORE_BASE_SPAN2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/Array2View.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace detail
{
// Pour indiquer que Span2<T>::view() retourne un Array2View
// et Span2<const T>::view() retourne un ConstArray2View.
template<typename T>
class View2TypeT
{
 public:
  using view_type = Array2View<T>;
};
template<typename T>
class View2TypeT<const T>
{
 public:
  using view_type = ConstArray2View<T>;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vue pour un tableau 2D.
 *
 * Comme toute vue, une instance de cette classe n'est valide que tant
 * que le conteneur dont elle est issue ne change pas de nombre d'éléments.
 */
template<class T>
class Span2
{
 public:

  using ElementType = T;
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using index_type = Int64;
  using difference_type = Int64;
  using pointer = ElementType*;
  using const_pointer = typename std::add_const<ElementType*>::type;
  using reference = ElementType&;
  using view_type = typename detail::View2TypeT<ElementType>::view_type;

 public:
  //! Créé une vue 2D de dimension [\a dim1_size][\a dim2_size]
  ARCCORE_HOST_DEVICE Span2(ElementType* ptr,Int64 dim1_size,Int64 dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size) {}
  //! Créé une vue 2D vide.
  ARCCORE_HOST_DEVICE Span2() : m_ptr(nullptr), m_dim1_size(0), m_dim2_size(0) {}
  //! Constructeur de recopie depuis une autre vue
  Span2(const Array2View<value_type>& from)
  : m_ptr(from.m_ptr), m_dim1_size(from.dim1Size()),m_dim2_size(from.dim2Size()) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  Span2(const ConstArray2View<X>& from)
  : m_ptr(from.data()), m_dim1_size(from.dim1Size()),m_dim2_size(from.dim2Size()) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span2(const Span2<X>& from)
  : m_ptr(from.data()), m_dim1_size(from.dim1Size()),m_dim2_size(from.dim2Size()) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
 public:
  //! Nombre d'éléments de la première dimension
  ARCCORE_HOST_DEVICE Int64 dim1Size() const { return m_dim1_size; }
  //! Nombre d'éléments de la deuxième dimension
  ARCCORE_HOST_DEVICE Int64 dim2Size() const { return m_dim2_size; }
  //! Nombre total d'éléments.
  ARCCORE_HOST_DEVICE Int64 totalNbElement() const { return m_dim1_size*m_dim2_size; }
 public:
  ARCCORE_HOST_DEVICE Span<ElementType> operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return Span<ElementType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }
  //! Valeur de l'élément [\a i][\a j]
  ARCCORE_HOST_DEVICE ElementType item(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    ARCCORE_CHECK_AT(j,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
  //! Positionne l'élément [\a i][\a j] à \a value
  ARCCORE_HOST_DEVICE ElementType setItem(Int64 i,Int64 j,const ElementType& value)
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    ARCCORE_CHECK_AT(j,m_dim2_size);
    m_ptr[(m_dim2_size*i) + j] = value;
  }
 public:
  /*!
   * \brief Vue constante sur cette vue.
   */
  view_type smallView()
  {
    Integer s1 = arccoreCheckArraySize(m_dim1_size);
    Integer s2 = arccoreCheckArraySize(m_dim2_size);
    return view_type(m_ptr,s1,s2);
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  ConstArrayView<value_type> constSmallView() const
  {
    Integer s1 = arccoreCheckArraySize(m_dim1_size);
    Integer s2 = arccoreCheckArraySize(m_dim2_size);
    return ConstArrayView<value_type>(m_ptr,s1,s2);
  }
 public:
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  inline ElementType* unguardedBasePointer() { return m_ptr; }
  /*!
   * \brief Pointeur sur la mémoire allouée.
   */
  ARCCORE_HOST_DEVICE inline ElementType* data() { return m_ptr; }
  /*!
   * \brief Pointeur constant sur la mémoire allouée.
   */
  ARCCORE_HOST_DEVICE inline const ElementType* data() const { return m_ptr; }
 private:
  ElementType* m_ptr;
  Int64 m_dim1_size;
  Int64 m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
