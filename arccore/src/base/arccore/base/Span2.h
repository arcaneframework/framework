// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Span2.h                                                     (C) 2000-2025 */
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

namespace Arcane
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
 * La vue est non modifiable si l'argument template est de type 'const T'.
 * Cette classe permet d'accéder et d'utiliser un tableau d'éléments du
 * type \a T de la même manière qu'un tableau C standard. \a SizeType est le
 * type utilisé pour conserver le nombre d'éléments du tableau. Cela peut
 * être 'Int32' ou 'Int64'.
 */
template<typename T,typename SizeType,SizeType Extent1,SizeType Extent2>
class Span2Impl
{
  using ThatClass = Span2Impl<T,SizeType,Extent1,Extent2>;

 public:

  using ElementType = T;
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using index_type = SizeType;
  using difference_type = SizeType;
  using size_type = SizeType;
  using pointer = ElementType*;
  using const_pointer = typename std::add_const<ElementType*>::type;
  using reference = ElementType&;
  using const_reference = const ElementType&;
  using view_type = typename detail::View2TypeT<ElementType>::view_type;

  //! Indique si on peut convertir un 'X' ou 'const X' en un 'T'
  template<typename X>
  using is_same_const_type = std::enable_if_t<std::is_same_v<X,T> || std::is_same_v<std::add_const_t<X>,T>>;

 public:

  //! Créé une vue 2D de dimension [\a dim1_size][\a dim2_size]
  ARCCORE_HOST_DEVICE Span2Impl(pointer ptr,SizeType dim1_size,SizeType dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size) {}
  //! Créé une vue 2D vide.
  ARCCORE_HOST_DEVICE Span2Impl() : m_ptr(nullptr), m_dim1_size(0), m_dim2_size(0) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  Span2Impl(const ConstArray2View<X>& from)
  : m_ptr(from.data()), m_dim1_size(from.dim1Size()),m_dim2_size(from.dim2Size()) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span2Impl(const Span2<X>& from)
  : m_ptr(from.data()), m_dim1_size(from.dim1Size()),m_dim2_size(from.dim2Size()) {}

 public:

  //! Nombre d'éléments de la première dimension
  constexpr ARCCORE_HOST_DEVICE SizeType dim1Size() const { return m_dim1_size; }
  //! Nombre d'éléments de la deuxième dimension
  constexpr ARCCORE_HOST_DEVICE SizeType dim2Size() const { return m_dim2_size; }
  //! Nombre total d'éléments.
  constexpr ARCCORE_HOST_DEVICE SizeType totalNbElement() const { return m_dim1_size*m_dim2_size; }

 public:

  constexpr ARCCORE_HOST_DEVICE SpanImpl<ElementType,SizeType> operator[](SizeType i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return SpanImpl<ElementType,SizeType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }

  constexpr ARCCORE_HOST_DEVICE SpanImpl<ElementType,SizeType> operator()(SizeType i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return SpanImpl<ElementType,SizeType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }

  constexpr ARCCORE_HOST_DEVICE reference operator()(SizeType i,SizeType j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }

#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  constexpr ARCCORE_HOST_DEVICE reference operator[](SizeType i,SizeType j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
#endif

  //! Valeur de l'élément [\a i][\a j]
  constexpr ARCCORE_HOST_DEVICE ElementType item(SizeType i,SizeType j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }

  //! Positionne l'élément [\a i][\a j] à \a value
  constexpr ARCCORE_HOST_DEVICE ElementType setItem(SizeType i,SizeType j,const ElementType& value)
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    m_ptr[(m_dim2_size*i) + j] = value;
  }

 public:

  /*!
   * \brief Vue constante sur cette vue.
   */
  constexpr view_type smallView()
  {
    Integer s1 = arccoreCheckArraySize(m_dim1_size);
    Integer s2 = arccoreCheckArraySize(m_dim2_size);
    return view_type(m_ptr,s1,s2);
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  constexpr ConstArrayView<value_type> constSmallView() const
  {
    Integer s1 = arccoreCheckArraySize(m_dim1_size);
    Integer s2 = arccoreCheckArraySize(m_dim2_size);
    return ConstArrayView<value_type>(m_ptr,s1,s2);
  }

 public:

  //! Pointeur sur la mémoire allouée.
  constexpr ElementType* unguardedBasePointer() { return m_ptr; }

  //! Pointeur sur la mémoire allouée.
  constexpr ARCCORE_HOST_DEVICE ElementType* data() { return m_ptr; }

  //! Pointeur sur la mémoire allouée.
  constexpr ARCCORE_HOST_DEVICE const ElementType* data() const { return m_ptr; }

 public:

  //! Opérateur d'égalité (valide si T est const mais pas X)
  template<typename X,SizeType XExtent1,SizeType XExtent2, typename = std::enable_if_t<std::is_same_v<X,value_type>>>
  friend bool operator==(const ThatClass& lhs, const Span2Impl<X,SizeType,XExtent1,XExtent2>& rhs)
  {
    return impl::areEqual2D(rhs,lhs);
  }
  //! Opérateur d'inégalité (valide si T est const mais pas X)
  template<typename X,SizeType XExtent1,SizeType XExtent2, typename = std::enable_if_t<std::is_same_v<X,value_type>>>
  friend bool operator!=(const ThatClass& lhs, const Span2Impl<X,SizeType,XExtent1,XExtent2>& rhs)
  {
    return !impl::areEqual2D(rhs,lhs);
  }
  //! Opérateur d'égalité
  template<SizeType XExtent1,SizeType XExtent2>
  friend bool operator==(const ThatClass& lhs, const Span2Impl<T,SizeType,XExtent1,XExtent2>& rhs)
  {
    return impl::areEqual2D(rhs,lhs);
  }
  //! Opérateur d'inégalité
  template<SizeType XExtent1,SizeType XExtent2>
  friend bool operator!=(const ThatClass& lhs, const Span2Impl<T,SizeType,XExtent1,XExtent2>& rhs)
  {
    return !impl::areEqual2D(rhs,lhs);
  }

 protected:

  ElementType* m_ptr;
  SizeType m_dim1_size;
  SizeType m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vue pour un tableau 2D dont la taille est un 'Int32'
 *
 * Comme toute vue, une instance de cette classe n'est valide que tant
 * que le conteneur dont elle est issue ne change pas de nombre d'éléments.
 */
template<class T,Int32 Extent1,Int32 Extent2>
class SmallSpan2
: public Span2Impl<T,Int32,Extent1,Extent2>
{
  friend class Span2<T>;

 public:

  using ThatClass = SmallSpan2<T,Extent1,Extent2>;
  using BaseClass = Span2Impl<T,Int32,Extent1,Extent2>;
  using size_type = Int32;
  using value_type = typename BaseClass::value_type;
  using pointer = typename BaseClass::pointer;
  using BaseClass::operator();
  using BaseClass::operator[];
  using ElementType = typename BaseClass::ElementType;

 private:

  using BaseClass::m_ptr;
  using BaseClass::m_dim1_size;
  using BaseClass::m_dim2_size;

 public:

  //! Créé une vue 2D de dimension [\a dim1_size][\a dim2_size]
  ARCCORE_HOST_DEVICE SmallSpan2(pointer ptr,Int32 dim1_size,Int32 dim2_size)
  : BaseClass(ptr,dim1_size,dim2_size) {}
  //! Créé une vue 2D vide.
  ARCCORE_HOST_DEVICE SmallSpan2() : BaseClass() {}
  //! Constructeur de recopie depuis une autre vue
  SmallSpan2(const Array2View<value_type>& from)
  : BaseClass(from.m_ptr,from.dim1Size(),from.dim2Size()) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  SmallSpan2(const ConstArray2View<X>& from)
  : BaseClass(from.m_ptr,from.dim1Size(),from.dim2Size()) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE SmallSpan2(const SmallSpan2<X>& from)
  : BaseClass(from.data(),from.dim1Size(),from.dim2Size()) {}

 public:

  ARCCORE_HOST_DEVICE SmallSpan<ElementType> operator[](Int32 i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return SmallSpan<ElementType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }

  ARCCORE_HOST_DEVICE SmallSpan<ElementType> operator()(Int32 i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return SmallSpan<ElementType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vue pour un tableau 2D dont la taille est un 'Int64'
 *
 * Comme toute vue, une instance de cette classe n'est valide que tant
 * que le conteneur dont elle est issue ne change pas de nombre d'éléments.
 */
template<class T,Int64 Extent1,Int64 Extent2>
class Span2
: public Span2Impl<T,Int64,Extent1,Extent2>
{
 public:

  using ThatClass = Span2<T,Extent1,Extent2>;
  using BaseClass = Span2Impl<T,Int64,Extent1,Extent2>;
  using size_type = Int64;
  using value_type = typename BaseClass::value_type;
  using pointer = typename BaseClass::pointer;
  using BaseClass::operator();
  using BaseClass::operator[];
  using ElementType = typename BaseClass::ElementType;

 private:

  using BaseClass::m_ptr;
  using BaseClass::m_dim1_size;
  using BaseClass::m_dim2_size;

 public:

  //! Créé une vue 2D de dimension [\a dim1_size][\a dim2_size]
  ARCCORE_HOST_DEVICE Span2(pointer ptr,Int64 dim1_size,Int64 dim2_size)
  : BaseClass(ptr,dim1_size,dim2_size) {}
  //! Créé une vue 2D vide.
  ARCCORE_HOST_DEVICE Span2() : BaseClass() {}
  //! Constructeur de recopie depuis une autre vue
  Span2(const Array2View<value_type>& from)
  : BaseClass(from.m_ptr,from.dim1Size(),from.dim2Size()) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  Span2(const ConstArray2View<X>& from)
  : BaseClass(from.m_ptr,from.dim1Size(),from.dim2Size()) {}

  //! Constructeur de recopie depuis un 'SmallSpan'
  Span2(const SmallSpan2<T>& from)
  : BaseClass(from.m_ptr,from.dim1Size(),from.dim2Size()) {}

  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span2(const Span2<X>& from)
  : BaseClass(from) {}

  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span2(const SmallSpan2<X>& from)
  : BaseClass(from.data(), from.dim1Size(), from.dim2Size()) {}

 public:

  ARCCORE_HOST_DEVICE Span<ElementType> operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return Span<ElementType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }

  ARCCORE_HOST_DEVICE Span<ElementType> operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return Span<ElementType>(m_ptr + (m_dim2_size*i),m_dim2_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
