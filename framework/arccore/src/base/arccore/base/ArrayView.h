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
/* ArrayView.h                                                 (C) 2000-2019 */
/*                                                                           */
/* Types définissant les vues de tableaux C.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEW_H
#define ARCCORE_BASE_ARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayRange.h"

#include <iostream>
#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> class ConstArrayView;
template<typename T> class ConstIterT;
template<typename T> class IterT;
template<typename T> class Span;
template<typename T> class SmallSpan;

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
  friend class Span<T>;
  friend class Span<const T>;
  friend class SmallSpan<T>;
  friend class SmallSpan<const T>;
 public:

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


 public:

  //! Construit une vue vide.
  ArrayView() : m_size(0), m_ptr(0) {}
  //! Constructeur de recopie depuis une autre vue
  ArrayView(const ArrayView<T>& from)
  : m_size(from.m_size), m_ptr(from.m_ptr) {}
  //! Construit une vue sur une zone mémoire commencant par \a ptr et
  // contenant \a asize éléments.
  explicit ArrayView(Integer asize,T* ptr) : m_size(asize), m_ptr(ptr) {}
  //! Opérateur de recopie
  const ArrayView<T>& operator=(const ArrayView<T>& from)
    { m_size=from.m_size; m_ptr=from.m_ptr; return *this; }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline T& operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline void setItem(Integer i,const T& v)
  {
    ARCCORE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  inline Integer size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Integer length() const { return m_size; }

  //! Itérateur sur le premier élément du tableau.
  iterator begin() { return iterator(m_ptr); }
  //! Itérateur sur le premier élément après la fin du tableau.
  iterator end() { return iterator(m_ptr+m_size); }
  //! Itérateur constant sur le premier élément du tableau.
  const_iterator begin() const { return const_iterator(m_ptr); }
  //! Itérateur constant sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_ptr+m_size); }
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
  T* ptrAt(Integer index)
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  //! Addresse du index-ème élément
  const T* ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  const T& at(Integer i) const
  {
    arccoreCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Integer i,const T& value)
  {
    arccoreCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  void fill(T o)
  {
    for( Integer i=0, n=m_size; i<n; ++i )
      m_ptr[i] = o;
  }

  /*!
   * \brief Vue constante sur cette vue.
   */
  ConstArrayView<T> constView() const
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
  ArrayView<T> subView(Integer abegin,Integer asize)
  {
    if (abegin>=m_size)
      return ArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue constante à partir de
   * l'élément `abegin` et contenant `asize` éléments.
   *
   * Si (\a abegin+ \a asize) est supérieur à la taille du tableau,
   * la vue est tronquée à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Integer abegin,Integer asize) const
  {
    if (abegin>=m_size)
      return ConstArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstArrayView<T>(asize,m_ptr+abegin);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ArrayView<T> subViewInterval(Integer index,Integer nb_interval)
  {
    Integer n = m_size;
    Integer isize = n / nb_interval;
    Integer ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    return ArrayView<T>(isize,m_ptr+ibegin);
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
    auto copy_size = copy_array.size();
    const T* copy_begin = copy_array.data();
    T* to_ptr = m_ptr;
    Integer n = m_size;
    if (copy_size<m_size)
      n = (Integer)copy_size;
    for( Integer i=0; i<n; ++i )
      to_ptr[i] = copy_begin[i];
  }

  //! Retourne \a true si le tableau est vide (dimension nulle)
  bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const T& v) const
  {
    for( Integer i=0; i<m_size; ++i ){
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

  /*!
   * \brief Pointeur sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  T* unguardedBasePointer() { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  const T* unguardedBasePointer() const { return m_ptr; }

  /*!
   * \brief Pointeur sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  const T* data() const { return m_ptr; }

  /*!
   * \brief Pointeur constant sur le début de la vue.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  T* data() { return m_ptr; }

 protected:

  /*!
   * \brief Retourne un pointeur sur le tableau.
   *
   * Cette méthode est identique à unguardedBasePointer() (i.e: il faudra
   * penser à la supprimer)
   */
  T* _ptr() { return m_ptr; }
  /*!
   * \brief Retourne un pointeur sur le tableau
   *
   * Cette méthode est identique à unguardedBasePointer() (i.e: il faudra
   * penser à la supprimer)
   */
  const T* _ptr() const { return m_ptr; }
  
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setArray(T* v,Integer s){ m_ptr = v; m_size = s; }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setPtr(T* v) { m_ptr = v; }

  /*!
   * \brief Modifie la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  void _setSize(Integer s) { m_size = s; }

 private:

  Integer m_size; //!< Nombre d'éléments du tableau
  T* m_ptr;  //!< Pointeur sur le tableau

 private:

  static Integer _min(Integer a,Integer b)
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

  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ConstArrayView<T> > const_iter;

  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Construit un tableau vide.
  ConstArrayView() : m_size(0), m_ptr(nullptr) {}
  //! Construit un tableau avec \a s élément
  explicit ConstArrayView(Integer s,const T* ptr)
  : m_size(s), m_ptr(ptr) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstArrayView(const ConstArrayView<T>& from)
  : m_size(from.m_size), m_ptr(from.m_ptr) {}
  /*! \brief Constructeur par copie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  ConstArrayView(const ArrayView<T>& from)
  : m_size(from.size()), m_ptr(from.data()) { }

  /*!
   * \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstArrayView<T>& operator=(const ConstArrayView<T>& from)
  { m_size=from.m_size; m_ptr=from.m_ptr; return *this; }

  /*! \brief Opérateur de recopie.
   * \warning Seul le pointeur est copié. Aucune copie mémoire n'est effectuée.
   */
  const ConstArrayView<T>& operator=(const ArrayView<T>& from)
  {
    m_size = from.size();
    m_ptr  = from.data();
    return (*this);
  }

 public:

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subView(Integer abegin,Integer asize) const
  {
    if (abegin>=m_size)
      return ConstArrayView<T>();
    asize = _min(asize,m_size-abegin);
    return ConstArrayView<T>(asize,m_ptr+abegin);
  }

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a abegin et
   * contenant \a asize éléments.
   *
   * Si `(abegin+asize)` est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Integer abegin,Integer asize) const
  {
    return subView(abegin,asize);
  }

  //! Sous-vue correspondant à l'interval \a index sur \a nb_interval
  ArrayView<T> subViewInterval(Integer index,Integer nb_interval)
  {
    Integer n = m_size;
    Integer isize = n / nb_interval;
    Integer ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index+1)==nb_interval)
      isize = n - ibegin;
    ARCCORE_CHECK_AT(ibegin+isize,n);
    return ConstArrayView<T>(isize,m_ptr+ibegin);
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T& operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode `check`, vérifie les débordements.
   */
  inline const T& item(Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  //! Nombre d'éléments du tableau
  inline Integer size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Integer length() const { return m_size; }
  //! Itérateur sur le premier élément du tableau.
  const_iterator begin() const { return const_iterator(m_ptr); }
  //! Itérateur sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_ptr+m_size); }
  //! Itérateur inverse sur le premier élément du tableau.
  const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }
  //! Itérateur inverse sur le premier élément après la fin du tableau.
  const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }
  //! \a true si le tableau est vide (size()==0)
  inline bool empty() const { return m_size==0; }
  //! \a true si le tableau contient l'élément de valeur \a v
  bool contains(const T& v) const
  {
    for( Integer i=0; i<m_size; ++i ){
      if (m_ptr[i]==v)
        return true;
    }
    return false;
  }
  void setArray(const ConstArrayView<T>& v) { m_ptr = v.m_ptr; m_size = v.m_size; }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  inline const T* unguardedBasePointer() const
  { return m_ptr; }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Les accès via le pointeur retourné ne pourront pas être
   * pas vérifiés par Arcane à la différence des accès via
   * operator[](): aucune vérification de dépassement n'est possible,
   * même en mode vérification.
   */
  const T* data() const
  { return m_ptr; }
  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_size);
  }

 private:

  Integer m_size; //!< Nombre d'éléments 
  const T* m_ptr; //!< Pointeur sur le début du tableau

 private:

  static Integer _min(Integer a,Integer b)
  {
    return ( (a<b) ? a : b );
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(ConstArrayView<T> rhs, ConstArrayView<T> lhs)
{
  if (rhs.size()!=lhs.size())
    return false;
  Integer s = rhs.size();
  for( Integer i=0; i<s; ++i ){
    if (rhs[i]==lhs[i])
      continue;
    else
      return false;
  }
  return true;
}

template<typename T> inline bool
operator!=(ConstArrayView<T> rhs, ConstArrayView<T> lhs)
{
  return !(rhs==lhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(ArrayView<T> rhs, ArrayView<T> lhs)
{
  return operator==(rhs.constView(),lhs.constView());
}

template<typename T> inline bool
operator!=(ArrayView<T> rhs,ArrayView<T> lhs)
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
dumpArray(std::ostream& o,ConstArrayView<T> val,int max_print)
{
  Integer n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) dernièrs
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    Integer z = (max_print/2);
    Integer z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( Integer i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( Integer i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( Integer i=0; i<n; ++i ){
      if (i!=0)
        o << ' ';
      o << "[" << i << "]=\"" << val[i] << '"';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, ConstArrayView<T> val)
{
  dumpArray(o,val,500);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, ArrayView<T> val)
{
  o << val.constView();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned int size);

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Integer' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Integer'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(int size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vérifie que \a size peut être converti dans un 'Int64' pour servir
 * de taille à un tableau.
 * Si possible, retourne \a size convertie en un 'Int64'. Sinon, lance
 * une exception de type ArgumentException.
 */
extern "C++" ARCCORE_BASE_EXPORT Int64
arccoreCheckLargeArraySize(size_t size);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
