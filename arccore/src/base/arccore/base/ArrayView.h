// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Types définissant les vues de tableaux C.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEW_H
#define ARCCORE_BASE_ARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayRange.h"
#include "arccore/base/ArrayViewCommon.h"
#include "arccore/base/BaseTypes.h"

#include <cstddef>
#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> class ConstArrayView;
template<typename T> class ConstIterT;
template<typename T> class IterT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Vue modifiable d'un tableau d'un type \a T.
 *
 Cette classe template permet d'accéder et d'utiliser un tableau d'éléments du
 type \a T de la même manière qu'un tableau C standard. Elle maintient en
 plus la taille du tableau. La fonction size() permet de connaître le nombre
 d'éléments du tableau et l'opérateur operator[]() permet d'accéder à un élément donné.

 Il est garanti que tous les éléments de la vue sont consécutifs en mémoire.

 Cette classe ne gère aucune mémoire et c'est le conteneur associé qui la gère.
 Les conteneurs possibles fournis par %Arccore sont les classes Array,
 UniqueArray ou SharedArray. Une vue n'est valide que tant que le conteneur associé n'est pas réalloué.
 De même, le contructeur et l'opérateur de recopie ne font que recopier les pointeurs
 sans réallouer la mémoire. Il faut donc les utiliser avec précaution.
 
 Si %Arccore est compilé en mode vérification (ARCCORE_CHECK est défini), les accès
 par l'intermédiaire de l'opérateurs operator[]() sont vérifiés et une
 exception IndexOutOfRangeException est lancé si un débordement de
 tableau a lieu. En attachant une session de debug au processus, il est
 possible de voir la pile d'appel au moment du débordement.

 Voici des exemples d'utilisation:

 \code
 Real t[5];
 ArrayView<Real> a(t,5); // Gère un tableau de 5 réels.
 Integer i = 3;
 Real v = a[2]; // Affecte à la valeur du 2ème élément
 a[i] = 5; // Affecte au 3ème élément la valeur 5
 \endcode

 Il est aussi possible d'accéder aux éléments du tableau par des
 itérateurs de la même manière qu'avec les containers de la STL.

 L'exemple suivant créé un itérateur \e i sur le tableau \e a et itère
 sur tout le tableau (méthode i()) et affiche les éléments:

 \code
 * for( Real v : a )
 *   cout << v;
 \endcode

 L'exemple suivant fait la somme des 3 premiers éléments du tableau:

 \code
 * Real sum = 0.0;
 * for( Real v : a.subView(0,3) )
 *   sum += v;
 \endcode

*/
template<class T>
class ArrayView
{
  template <typename T2, Int64 Extent, Int64 MinValue> friend class Span;
  template <typename T2, Int32 Extent, Int32 MinValue> friend class SmallSpan;

 public:

  using ThatClass = ArrayView<T>;

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type de l'itérateur sur un élément du tableau
  typedef ArrayIterator<pointer> iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

  //! Type d'un itérateur sur tout le tableau
  typedef IterT< ArrayView<T> > iter;
  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ArrayView<T> > const_iter;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Construit une vue vide.
  constexpr ArrayView() noexcept : m_size(0), m_ptr(nullptr) {}

  //! Constructeur de recopie depuis une autre vue
  ArrayView(const ArrayView<T>& from) = default;

  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  constexpr ArrayView(Integer asize,pointer ptr)  noexcept : m_size(asize), m_ptr(ptr) {}

  //! Construit une vue sur une zone mémoire commencant par \a ptr et contenant \a asize éléments.
  template<std::size_t N>
  constexpr ArrayView(std::array<T,N>& v)
  : m_size(arccoreCheckArraySize(v.size())), m_ptr(v.data()) {}

  //! Opérateur de recopie
  ArrayView<T>& operator=(const ArrayView<T>& from) = default;

  template<std::size_t N>
  constexpr ArrayView<T>& operator=(std::array<T,N>& from)
  {
    m_size = arccoreCheckArraySize(from.size());
    m_ptr = from.data();
    return (*this);
  }

 public:

  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  static constexpr ThatClass create(pointer ptr,Integer asize) noexcept
  {
    return ThatClass(asize,ptr);
  }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr reference operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr const_reference operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr reference operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr const_reference operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr const_reference item(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr void setItem(Integer i,const_reference v)
  {
    ARCCORE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  constexpr Integer size() const noexcept { return m_size; }
  //! Nombre d'éléments du tableau
  constexpr Integer length() const noexcept { return m_size; }

  //! Itérateur sur le premier élément du tableau.
  constexpr iterator begin() noexcept { return iterator(m_ptr); }
  //! Itérateur sur le premier élément après la fin du tableau.
  constexpr iterator end() noexcept { return iterator(m_ptr+m_size); }
  //! Itérateur constant sur le premier élément du tableau.
  constexpr const_iterator begin() const noexcept { return const_iterator(m_ptr); }
  //! Itérateur constant sur le premier élément après la fin du tableau.
  constexpr const_iterator end() const noexcept { return const_iterator(m_ptr+m_size); }
  //! Itérateur inverse sur le premier élément du tableau.
  constexpr reverse_iterator rbegin() noexcept { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément du tableau.
  constexpr const_reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  constexpr reverse_iterator rend() noexcept { return std::make_reverse_iterator(begin()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  constexpr const_reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }

 public:

  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr,m_ptr+m_size);
  }
  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 public:
  //! Addresse du index-ème élément
  constexpr pointer ptrAt(Integer index)
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  //! Addresse du index-ème élément
  constexpr const_pointer ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  constexpr const_reference at(Integer i) const
  {
    arccoreCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Integer i,const_reference value)
  {
    arccoreCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  void fill(const T& o) noexcept
  {
    for( Integer i=0, n=m_size; i<n; ++i )
      m_ptr[i] = o;
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  constexpr ConstArrayView<T> constView() const noexcept
  {
    return ConstArrayView<T>(m_size,m_ptr);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ArrayView<T> subView(Integer abegin,Integer asize) noexcept
  {
    if (abegin>=m_size)
      return ArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a abegin
   * et contenant \a asize éléments.
   *
   * Si (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ThatClass subPart(Integer abegin,Integer asize) noexcept
  {
    return subView(abegin,asize);
  }

  /*!
   * \brief Sous-vue constante à partir de
   * l'élément `abegin` et contenant `asize` éléments.
   *
   * Si (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ConstArrayView<T> subConstView(Integer abegin,Integer asize) const noexcept
  {
    if (abegin>=m_size)
      return ConstArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstArrayView<T>(asize,m_ptr+abegin);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  constexpr ArrayView<T> subViewInterval(Integer index,Integer nb_interval)
  {
    return impl::subViewInterval<ThatClass>(*this,index,nb_interval);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  constexpr ThatClass subPartInterval(Integer index,Integer nb_interval)
  {
    return impl::subViewInterval<ThatClass>(*this,index,nb_interval);
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
  void copy(const U& copy_array)
  {
    auto copy_size = copy_array.size();
    const_pointer copy_begin = copy_array.data();
    pointer to_ptr = m_ptr;
    Integer n = m_size;
    if (copy_size<m_size)
      n = (Integer)copy_size;
    for( Integer i=0; i<n; ++i )
      to_ptr[i] = copy_begin[i];
  }

  //! Retourne \a true si le tableau est vide (dimension nulle)
  constexpr bool empty() const noexcept { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const_reference v) const
  {
    for( Integer i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }

 public:

  void setArray(const ArrayView<T>& v) noexcept
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
  constexpr pointer unguardedBasePointer() noexcept { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  constexpr const_pointer unguardedBasePointer() const noexcept { return m_ptr; }

  /*!
   * \brief Pointeur sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  constexpr const_pointer data() const noexcept { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  constexpr pointer data() noexcept { return m_ptr; }

 public:

  friend inline bool operator==(const ArrayView<T>& rhs, const ArrayView<T>& lhs)
  {
    return impl::areEqual(rhs,lhs);
  }

  friend inline bool operator!=(const ArrayView<T>& rhs, const ArrayView<T>& lhs)
  {
    return !(rhs==lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const ArrayView<T>& val)
  {
    impl::dumpArray(o,val,500);
    return o;
  }

 protected:

  /*!
   * \brief Retourne un pointeur sur le tableau.
   *
   * Cette méthode est identique à unguardedBasePointer() (i.e: il faudra
   * penser à la supprimer)
   */
  constexpr pointer _ptr() noexcept { return m_ptr; }
  /*!
   * \brief Retourne un pointeur sur le tableau
   *
   * Cette méthode est identique à unguardedBasePointer() (i.e: il faudra
   * penser à la supprimer)
   */
  constexpr const_pointer _ptr() const noexcept { return m_ptr; }
  
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setArray(pointer v,Integer s) noexcept { m_ptr = v; m_size = s; }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setPtr(pointer v) noexcept { m_ptr = v; }

  /*!
   * \brief Modifie la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setSize(Integer s) noexcept { m_size = s; }

 private:

  Integer m_size; //!< Nombre d'éléments du tableau
  pointer m_ptr;  //!< Pointeur sur le tableau

 private:

  static constexpr Integer _min(Integer a,Integer b) noexcept
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection 
 * \brief Vue constante d'un tableau de type \a T.
 *
 * Cette classe fonctionne de la même manière que ArrayView à la seule
 * différence qu'il n'est pas possible de modifier les éléments du tableau.
 */
template<class T>
class ConstArrayView
{
  friend class Span<T>;
  friend class Span<const T>;
  friend class SmallSpan<T>;
  friend class SmallSpan<const T>;

 public:

  using ThatClass = ConstArrayView<T>;

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef std::ptrdiff_t difference_type;

  using const_value_type = typename std::add_const_t<value_type>;

  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ConstArrayView<T> > const_iter;

  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Construit un tableau vide.
  constexpr ConstArrayView() noexcept : m_size(0), m_ptr(nullptr) {}
  //! Construit un tableau avec \a s élément
  constexpr ConstArrayView(Integer s,const_pointer ptr) noexcept
  : m_size(s), m_ptr(ptr) {}
  //! Constructeur par copie.
  ConstArrayView(const ConstArrayView<T>& from) = default;
  /*!
   * \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  constexpr ConstArrayView(const ArrayView<T>& from) noexcept
  : m_size(from.size()), m_ptr(from.data()) { }

  //! Création depuis un std::array
  template<std::size_t N,typename X,typename = std::enable_if_t<std::is_same_v<X,const_value_type>> >
  constexpr ConstArrayView(const std::array<X,N>& v)
  : m_size(arccoreCheckArraySize(v.size())), m_ptr(v.data()) {}

  /*!
   * \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstArrayView<T>& operator=(const ConstArrayView<T>& from) = default;

  /*!
   * \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  constexpr ConstArrayView<T>& operator=(const ArrayView<T>& from)
  {
    m_size = from.size();
    m_ptr  = from.data();
    return (*this);
  }

  //! Opérateur de recopie
  template<std::size_t N,typename X,typename = std::enable_if_t<std::is_same_v<X,const_value_type>> >
  constexpr ConstArrayView<T>& operator=(const std::array<X,N>& from)
  {
    m_size = arccoreCheckArraySize(from.size());
    m_ptr = from.data();
    return (*this);
  }
 
 public:

  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  static constexpr ThatClass create(const_pointer ptr,Integer asize) noexcept
  {
    return ThatClass(asize,ptr);
  }

 public:

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ConstArrayView<T> subView(Integer abegin,Integer asize) const noexcept
  {
    if (abegin>=m_size)
      return ConstArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ThatClass subPart(Integer abegin,Integer asize) const noexcept
  {
    return subView(abegin,asize);
  }

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   * contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  constexpr ConstArrayView<T> subConstView(Integer abegin,Integer asize) const noexcept
  {
    return subView(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  constexpr ConstArrayView<T> subViewInterval(Integer index,Integer nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this,index,nb_interval);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  constexpr ThatClass subPartInterval(Integer index,Integer nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this,index,nb_interval);
  }

  //! Addresse du index-ème élément
  constexpr const_pointer ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr const_reference operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  constexpr const_reference operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode `check`, vérifie les débordements.
   */
  constexpr const_reference item(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  //! Nombre d'éléments du tableau
  constexpr Integer size() const noexcept { return m_size; }
  //! Nombre d'éléments du tableau
  constexpr Integer length() const noexcept { return m_size; }
  //! Itérateur sur le premier élément du tableau.
  constexpr const_iterator begin() const noexcept { return const_iterator(m_ptr); }
  //! Itérateur sur le premier élément après la fin du tableau.
  constexpr const_iterator end() const noexcept { return const_iterator(m_ptr+m_size); }
  //! Itérateur inverse sur le premier élément du tableau.
  constexpr const_reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  constexpr const_reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
  //! \a true si le tableau est vide (size()==0)
  constexpr bool empty() const noexcept { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const_reference v) const
  {
    for( Integer i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }
  void setArray(const ConstArrayView<T>& v) noexcept
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  constexpr const_pointer unguardedBasePointer() const noexcept { return m_ptr; }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  constexpr const_pointer data() const noexcept { return m_ptr; }

  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 public:

  friend inline bool operator==(const ConstArrayView<T>& rhs, const ConstArrayView<T>& lhs)
  {
    return Arcane::impl::areEqual(rhs,lhs);
  }

  friend inline bool operator!=(const ConstArrayView<T>& rhs, const ConstArrayView<T>& lhs)
  {
    return !(rhs==lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const ConstArrayView<T>& val)
  {
    Arcane::impl::dumpArray(o,val,500);
    return o;
  }

 private:

  Integer m_size; //!< Nombre d'éléments 
  const_pointer m_ptr; //!< Pointeur sur le début du tableau

 private:

  static constexpr Integer _min(Integer a,Integer b) noexcept
  {
    return ( (a<b) ? a : b );
  }
};

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
dumpArray(std::ostream& o,ConstArrayView<T> val,int max_print)
{
  impl::dumpArray(o,val,max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
