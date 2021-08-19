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
/* Span.h                                                      (C) 2000-2021 */
/*                                                                           */
/* Vues sur des tableaux C.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_SPAN_H
#define ARCCORE_BASE_SPAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace detail
{
// Pour indiquer que Span<T>::view() retourne un ArrayView
// et Span<const T>::view() retourn un ConstArrayView.
template<typename T>
class ViewTypeT
{
 public:
  using view_type = ArrayView<T>;
};
template<typename T>
class ViewTypeT<const T>
{
 public:
  using view_type = ConstArrayView<T>;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue d'un tableau d'éléments de type \a T.
 *
 * Cette classe ne doit pas être utilisée directement. Il faut utiliser
 * Span ou SmallSpan à la place.
 *
 * La vue est non modifiable si l'argument template est de type 'const T'.
 * Cette classe permet d'accéder et d'utiliser un tableau d'éléments du
 * type \a T de la même manière qu'un tableau C standard. \a SizeType est le
 * type utilisé pour conserver le nombre d'éléments du tableau. Cela peut
 * être 'Int32' ou 'Int64'.
*/
template<typename T,typename SizeType>
class SpanImpl
{
 public:

  using ThatClass = SpanImpl<T,SizeType>;
  using ElementType = T;
  using element_type = ElementType;
  using value_type = typename std::remove_cv<ElementType>::type;
  using index_type = SizeType;
  using difference_type = SizeType;
  using pointer = ElementType*;
  using const_pointer = typename std::add_const<ElementType*>::type ;
  using reference = ElementType&;
  using iterator = ArrayIterator<pointer>;
  using const_iterator = ArrayIterator<const_pointer>;
  using view_type = typename detail::ViewTypeT<ElementType>::view_type;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 public:

  //! Construit une vue vide.
  ARCCORE_HOST_DEVICE SpanImpl() : m_ptr(nullptr), m_size(0) {}
  //! Constructeur de recopie depuis une autre vue
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE SpanImpl(const SpanImpl<X,SizeType>& from)
  : m_ptr(from.data()), m_size(from.size()) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  ARCCORE_HOST_DEVICE SpanImpl(T* ptr,SizeType asize)
  : m_ptr(ptr), m_size(asize) {}

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  ARCCORE_HOST_DEVICE inline reference operator[](SizeType i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  ARCCORE_HOST_DEVICE inline reference item(SizeType i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  ARCCORE_HOST_DEVICE inline void setItem(SizeType i,const T& v)
  {
    ARCCORE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  ARCCORE_HOST_DEVICE inline SizeType size() const { return m_size; }
  //! Retourne la taille du tableau en octets
  ARCCORE_HOST_DEVICE inline Int64 sizeBytes() const { return m_size * sizeof(value_type); }
  //! Nombre d'éléments du tableau
  ARCCORE_HOST_DEVICE inline SizeType length() const { return m_size; }

  /*!
   * \brief Itérateur sur le premier élément du tableau.
   */
  ARCCORE_HOST_DEVICE iterator begin() const { return iterator(m_ptr); }
  /*!
   * \brief Itérateur sur le premier élément après la fin du tableau.
   */
  ARCCORE_HOST_DEVICE iterator end() const { return iterator(m_ptr+m_size); }
  //! Itérateur inverse sur le premier élément du tableau.
  ARCCORE_HOST_DEVICE reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  ARCCORE_HOST_DEVICE reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }

 public:

  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<pointer> range() const
  {
    return ArrayRange<pointer>(m_ptr,m_ptr+m_size);
  }

 public:

  //! Addresse du index-ème élément
  ARCCORE_HOST_DEVICE inline T* ptrAt(SizeType index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  ARCCORE_HOST_DEVICE reference at(SizeType i) const
  {
    arccoreCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  ARCCORE_HOST_DEVICE void setAt(SizeType i,const T& value)
  {
    arccoreCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  ARCCORE_HOST_DEVICE inline void fill(T o)
  {
    for( SizeType i=0, n=m_size; i<n; ++i )
      m_ptr[i] = o;
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  view_type smallView()
  {
    Integer s = arccoreCheckArraySize(m_size);
    return view_type(s,m_ptr);
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  ConstArrayView<value_type> constSmallView() const
  {
    Integer s = arccoreCheckArraySize(m_size);
    return ConstArrayView<value_type>(s,m_ptr);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ARCCORE_HOST_DEVICE ThatClass subspan(SizeType abegin,SizeType asize) const
  {
    if (abegin>=m_size)
      return {};
    asize = _min(asize,m_size-abegin);
    return {m_ptr+abegin,asize};
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ThatClass subView(SizeType abegin,SizeType asize) const
  {
    return subspan(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ThatClass subViewInterval(SizeType index,SizeType nb_interval) const
  {
    SizeType n = m_size;
    SizeType isize = n / nb_interval;
    SizeType ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    return {m_ptr+ibegin,isize};
  }

  /*!
   * \brief Recopie le tableau \a copy_array dans l'instance.
   *
   * Comme aucune allocation mémoire n'est effectuée, le
   * nombre d'éléments de \a copy_array doit être inférieur ou égal au
   * nombre d'éléments courant. S'il est inférieur, les éléments du
   * tableau courant situés à la fin du tableau sont inchangés
   */
  template<class U> ARCCORE_HOST_DEVICE
  inline void copy(const U& copy_array)
  {
    Int64 n = copy_array.size();
    Int64 size_as_int64 = m_size;
    arccoreCheckAt(n,size_as_int64+1);
    const T* copy_begin = copy_array.data();
    T* to_ptr = m_ptr;
    // On est sur que \a n tient sur un 'SizeType' car il est plus petit
    // que \a m_size
    SizeType n_as_sizetype = static_cast<SizeType>(n);
    for( SizeType i=0; i<n_as_sizetype; ++i )
      to_ptr[i] = copy_begin[i];
  }

  //! Retourne \a true si le tableau est vide (dimension nulle)
  ARCCORE_HOST_DEVICE bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  ARCCORE_HOST_DEVICE bool contains(const T& v) const
  {
    for( SizeType i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }

 public:

  ARCCORE_HOST_DEVICE void setArray(const ArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }
  ARCCORE_HOST_DEVICE void setArray(const Span<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointeur sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  ARCCORE_HOST_DEVICE pointer data() const { return m_ptr; }

 protected:
  
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setArray(T* v,SizeType s){ m_ptr = v; m_size = s; }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setPtr(T* v) { m_ptr = v; }

  /*!
   * \brief Modifie la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setSize(SizeType s) { m_size = s; }

 private:

  T* m_ptr;  //!< Pointeur sur le tableau
  SizeType m_size; //!< Nombre d'éléments du tableau

 private:

  static SizeType _min(SizeType a,SizeType b)
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue d'un tableau d'éléments de type \a T.
 *
 * La vue est non modifiable si l'argument template est de type 'const T'.
 Cette classe permet d'accéder et d'utiliser un tableau d'éléments du
 type \a T de la même manière qu'un tableau C standard. Elle est similaire à
 ArrayView à ceci près que le nombre d'éléments est stocké sur un 'Int64' et
 peut donc dépasser 2Go. Elle est concue pour être similaire à la classe
 std::span du C++20.
*/
template<typename T>
class Span
: public SpanImpl<T,Int64>
{
 public:

  using BaseClass = SpanImpl<T,Int64>;
  using value_type = typename BaseClass::value_type;

 public:

  //! Construit une vue vide.
  Span() = default;
  //! Constructeur de recopie depuis une autre vue
  ARCCORE_HOST_DEVICE Span(const ArrayView<value_type>& from)
  : BaseClass(from.m_ptr,from.m_size) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span(const ConstArrayView<X>& from)
  : BaseClass(from.m_ptr,from.m_size) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = std::enable_if_t<std::is_same_v<X,value_type>> >
  ARCCORE_HOST_DEVICE Span(const Span<X>& from)
  : BaseClass(from) {}
  ARCCORE_HOST_DEVICE Span(const SpanImpl<T,Int64>& from)
  : BaseClass(from) {}
  ARCCORE_HOST_DEVICE Span(const SpanImpl<T,Int32>& from)
  : BaseClass(from.data(),from.size()) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  ARCCORE_HOST_DEVICE Span(T* ptr,Int64 asize)
  : BaseClass(ptr,asize) {}

 public:

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ARCCORE_HOST_DEVICE Span<T> subspan(Int64 abegin,Int64 asize) const
  {
    return BaseClass::subspan(abegin,asize);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ARCCORE_HOST_DEVICE Span<T> subView(Int64 abegin,Int64 asize) const
  {
    return subspan(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ARCCORE_HOST_DEVICE Span<T> subViewInterval(Int64 index,Int64 nb_interval) const
  {
    return BaseClass::subViewInternal(index,nb_interval);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue d'un tableau d'éléments de type \a T.
 *
 * La vue est non modifiable si l'argument template est de type 'const T'.
 * Cette classe permet d'accéder et d'utiliser un tableau d'éléments du
 * type \a T de la même manière qu'un tableau C standard. Elle est similaire à
 * Span à ceci près que le nombre d'éléments est stocké sur un 'Int32'.
 */
template<typename T>
class SmallSpan
: public SpanImpl<T,Int32>
{
  // Pour le cas où on ne supporte pas le C++14.
  template< bool B, class XX = void >
  using Span_enable_if_t = typename std::enable_if<B,XX>::type;

 public:

  using BaseClass = SpanImpl<T,Int32>;
  using value_type = typename BaseClass::value_type;

 public:

  //! Construit une vue vide.
  SmallSpan() = default;
  //! Constructeur de recopie depuis une autre vue
  ARCCORE_HOST_DEVICE SmallSpan(const ArrayView<value_type>& from)
  : BaseClass(from.m_ptr,from.m_size) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = Span_enable_if_t<std::is_same<X,value_type>::value> >
  ARCCORE_HOST_DEVICE SmallSpan(const ConstArrayView<X>& from)
  : BaseClass(from.m_ptr,from.m_size) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = Span_enable_if_t<std::is_same<X,value_type>::value> >
  ARCCORE_HOST_DEVICE SmallSpan(const SmallSpan<X>& from)
  : BaseClass(from) {}
  ARCCORE_HOST_DEVICE SmallSpan(const SpanImpl<T,Int32>& from)
  : BaseClass(from) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  ARCCORE_HOST_DEVICE SmallSpan(T* ptr,Int32 asize)
  : BaseClass(ptr,asize) {}

 public:

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ARCCORE_HOST_DEVICE Span<T> subspan(Int64 abegin,Int64 asize) const
  {
    return BaseClass::subspan(abegin,asize);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ARCCORE_HOST_DEVICE Span<T> subView(Int64 abegin,Int64 asize) const
  {
    return subspan(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ARCCORE_HOST_DEVICE Span<T> subViewInterval(Int64 index,Int64 nb_interval) const
  {
    return BaseClass::subViewInternal(index,nb_interval);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T,typename SizeType> inline bool
operator==(SpanImpl<const T,SizeType> rhs, SpanImpl<const T,SizeType> lhs)
{
  if (rhs.size()!=lhs.size())
    return false;
  SizeType s = rhs.size();
  for( SizeType i=0; i<s; ++i ){
    if (rhs[i]==lhs[i])
      continue;
    else
      return false;
  }
  return true;
}

template<typename T> inline bool
operator==(Span<const T> rhs, Span<const T> lhs)
{
  SpanImpl<const T,Int64> a = rhs;
  SpanImpl<const T,Int64> b = lhs;
  return a==b;
}

template<typename T> inline bool
operator!=(Span<const T> rhs, Span<const T> lhs)
{
  return !(rhs==lhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(Span<T> rhs, Span<T> lhs)
{
  return operator==(rhs.constView(),lhs.constView());
}

template<typename T> inline bool
operator!=(Span<T> rhs, Span<T> lhs)
{
  return !(rhs==lhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(SmallSpan<T> rhs, SmallSpan<T> lhs)
{
  return operator==(Span<const T>(rhs),Span<const T>(lhs));
}

template<typename T> inline bool
operator!=(SmallSpan<T> rhs, SmallSpan<T> lhs)
{
  return !(rhs==lhs);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Affiche sur le flot \a o les valeurs du tableau \a val.
 *
 * Si \a max_print est positif, alors au plus \a max_print valeurs
 * sont affichées. Si la taille du tableau est supérieure à
 * \a max_print, alors les (max_print/2) premiers et derniers
 * éléments sont affichés.
 */
template<typename T,typename SizeType> inline void
dumpArray(std::ostream& o,SpanImpl<const T,SizeType> val,int max_print)
{
  SizeType n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) derniers
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    SizeType z = (max_print/2);
    SizeType z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( SizeType i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( SizeType i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( SizeType i=0; i<n; ++i ){
      if (i!=0)
        o << ' ';
      o << "[" << i << "]=\"" << val[i] << '"';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extrait un sous-tableau à à partir d'une liste d'index.
 *
 * Remplit \a result avec les valeurs du tableau \a values correspondantes
 * aux indices \a indexes.
 *
 * \pre results.size() >= indexes.size();
 */
template<typename DataType,typename IntegerType,typename SizeType> inline void
_sampleSpan(SpanImpl<const DataType,SizeType> values,
            SpanImpl<const IntegerType,SizeType> indexes,
            SpanImpl<DataType,SizeType> result)
{
  const Int64 result_size = indexes.size();
  [[maybe_unused]] const Int64 my_size = values.size();
  const DataType* ptr = values.data();
  for( Int64 i=0; i<result_size; ++i) {
    IntegerType index = indexes[i];
    ARCCORE_CHECK_AT(index,my_size);
    result[i] = ptr[index];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extrait un sous-tableau à à partir d'une liste d'index.
 *
 * Remplit \a result avec les valeurs du tableau \a values correspondantes
 * aux indices \a indexes.
 *
 * \pre results.size() >= indexes.size();
 */
template<typename DataType> inline void
sampleSpan(Span<const DataType> values,Span<const Int64> indexes,Span<DataType> result)
{
  _sampleSpan(values,indexes,result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Extrait un sous-tableau à à partir d'une liste d'index.
 *
 * Le résultat est stocké dans \a result dont la taille doit être au moins
 * égale à celle de \a indexes.
 */
template<typename DataType> inline void
sampleSpan(Span<const DataType> values,Span<const Int32> indexes,Span<DataType> result)
{
  _sampleSpan(values,indexes,result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la vue en un tableau d'octets non modifiables.
 */
template<typename DataType,typename SizeType> inline Span<const std::byte>
asBytes(SpanImpl<const DataType,SizeType> s)
{
  return {reinterpret_cast<const std::byte*>(s.data()), s.sizeBytes()};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti la vue en un tableau d'octets modifiables.
 *
 * Cette méthode n'est accessible que si \a DataType n'est pas `const`.
 */
template<typename DataType,typename SizeType,
         typename std::enable_if_t<!std::is_const<DataType>::value, int> = 0>
inline Span<std::byte>
asWritableBytes(SpanImpl<DataType,SizeType> s)
{
  return {reinterpret_cast<std::byte*>(s.data()), s.sizeBytes()};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, Span<const T> val)
{
  dumpArray(o,val,500);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, Span<T> val)
{
  o << Span<const T>(val);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, SmallSpan<const T> val)
{
  o << Span<const T>(val);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, SmallSpan<T> val)
{
  o << Span<const T>(val);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
