// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Span.h                                                      (C) 2000-2019 */
/*                                                                           */
/* Types définissant les vues de tableaux C dont la taille est un Int64.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_LARGEARRAYVIEW_H
#define ARCCORE_BASE_LARGEARRAYVIEW_H
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
 * La vue est non modifiable si l'argument template est de type 'const T'.
 Cette classe permet d'accéder et d'utiliser un tableau d'éléments du
 type \a T de la même manière qu'un tableau C standard. Elle est similaire à
 ArrayView à ceci près que le nombre d'éléments est stocké sur un 'Int64' et
 peut donc dépasser 2Go. Elle est concue pour être similaire à la classe
 std::span du C++20.
*/
template<typename T>
class Span
{
  // Pour le cas où on ne supporte pas le C++14.
  template< bool B, class XX = void >
  using Span_enable_if_t = typename std::enable_if<B,XX>::type;

 public:

  typedef T ElementType;
  typedef ElementType element_type;
  typedef typename std::remove_cv<ElementType>::type value_type;
  typedef Int64 index_type;
  typedef Int64 difference_type;
  typedef ElementType* pointer;
  typedef typename std::add_const<ElementType*>::type const_pointer;
  typedef ElementType& reference;
  typedef ArrayIterator<pointer> iterator;
  typedef ArrayIterator<const_pointer> const_iterator;
  typedef typename detail::ViewTypeT<ElementType>::view_type view_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Construit une vue vide.
  Span() : m_ptr(nullptr), m_size(0) {}
  //! Constructeur de recopie depuis une autre vue
  Span(const ArrayView<value_type>& from)
  : m_ptr(from.m_ptr), m_size(from.size()) {}
  // Constructeur à partir d'un ConstArrayView. Cela n'est autorisé que
  // si T est const.
  template<typename X,typename = Span_enable_if_t<std::is_same<X,value_type>::value> >
  Span(const ConstArrayView<X>& from)
  : m_ptr(from.data()), m_size(from.size()) {}
  // Pour un Span<const T>, on a le droit de construire depuis un Span<T>
  template<typename X,typename = Span_enable_if_t<std::is_same<X,value_type>::value> >
  Span(const Span<X>& from)
  : m_ptr(from.data()), m_size(from.size()) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  Span(T* ptr,Int64 asize)
  : m_ptr(ptr), m_size(asize) {}

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline void setItem(Int64 i,const T& v)
  {
    ARCCORE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  inline Int64 size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Int64 length() const { return m_size; }

  /*!
   * \brief Itérateur sur le premier élément du tableau.
   */
  iterator begin() { return iterator(m_ptr); }
  /*!
   * \brief Itérateur sur le premier élément après la fin du tableau.
   */
  iterator end() { return iterator(m_ptr+m_size); }
  /*!
   * \brief Itérateur constant sur le premier élément du tableau.
   */
  const_iterator begin() const { return iterator(m_ptr); }
  /*!
   * \brief Itérateur constant sur le premier élément après la fin du tableau.
   */
  const_iterator end() const { return iterator(m_ptr+m_size); }
  //! Itérateur inverse sur le premier élément du tableau.
  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément du tableau.
  const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }

 public:

  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr,m_ptr+m_size);
  }
  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 public:
  //! Addresse du index-ème élément
  inline T* ptrAt(Int64 index)
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Int64 index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  const T& at(Int64 i) const
  {
    arccoreCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Int64 i,const T& value)
  {
    arccoreCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  inline void fill(T o)
  {
    for( Int64 i=0, n=m_size; i<n; ++i )
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
  Span<T> subspan(Int64 abegin,Int64 asize) const
  {
    if (abegin>=m_size)
      return Span<T>();
    asize = _min(asize,m_size-abegin);
    return Span<T>(m_ptr+abegin,asize);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  Span<T> subView(Int64 abegin,Int64 asize) const
  {
    return subspan(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  Span<T> subViewInterval(Int64 index,Int64 nb_interval) const
  {
    Int64 n = m_size;
    Int64 isize = n / nb_interval;
    Int64 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    return Span<T>(m_ptr+ibegin,isize);
  }

  /*!
   * \brief Recopie le tableau \a copy_array dans l'instance.
   *
   * Comme aucune allocation mémoire n'est effectuée, le
   * nombre d'éléments de \a copy_array doit être inférieur ou égal au
   * nombre d'éléments courant. S'il est inférieur, les éléments du
   * tableau courant situés à la fin du tableau sont inchangés
   */
  template<class U>
  inline void copy(const U& copy_array)
  {
    ARCCORE_ASSERT( (copy_array.size()<=m_size), ("Bad size %d %d",copy_array.size(),m_size) );
    const T* copy_begin = copy_array.data();
    T* to_ptr = m_ptr;
    Int64 n = copy_array.size();
    for( Int64 i=0; i<n; ++i )
      to_ptr[i] = copy_begin[i];
  }

  //! Retourne \a true si le tableau est vide (dimension nulle)
  bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const T& v) const
  {
    for( Int64 i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }

 public:

  void setArray(const ArrayView<T>& v)
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }
  void setArray(const Span<T>& v)
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
  const_pointer data() const
  { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  pointer data()
  { return m_ptr; }

 protected:
  
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setArray(T* v,Int64 s){ m_ptr = v; m_size = s; }

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
  inline void _setSize(Int64 s) { m_size = s; }

 private:

  T* m_ptr;  //!< Pointeur sur le tableau
  Int64 m_size; //!< Nombre d'éléments du tableau

 private:

  static Int64 _min(Int64 a,Int64 b)
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(Span<const T> rhs, Span<const T> lhs)
{
  if (rhs.size()!=lhs.size())
    return false;
  Int64 s = rhs.size();
  for( Int64 i=0; i<s; ++i ){
    if (rhs[i]==lhs[i])
      continue;
    else
      return false;
  }
  return true;
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
/*!
 * \brief Affiche sur le flot \a o les valeurs du tableau \a val.
 *
 * Si \a max_print est positif, alors au plus \a max_print valeurs
 * sont affichées. Si la taille du tableau est supérieure à
 * \a max_print, alors les (max_print/2) premiers et derniers
 * éléments sont affichés.
 */
template<typename T> inline void
dumpArray(std::ostream& o,Span<const T> val,int max_print)
{
  Int64 n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) derniers
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    Int64 z = (max_print/2);
    Int64 z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( Int64 i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( Int64 i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( Int64 i=0; i<n; ++i ){
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
template<typename DataType,typename IntegerType> inline void
_sampleSpan(Span<const DataType> values,Span<const IntegerType> indexes,Span<DataType> result)
{
  const Int64 result_size = indexes.size();
  const Int64 my_size = values.size();
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
