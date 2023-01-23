// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Types définissant les vues de tableaux C.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYVIEW_H
#define ARCANE_UTILS_ARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> class ConstArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue d'un tableau d'un type \a T.
 *
 * \ingroup Collection
 *
 Cette classe template permet d'accéder et d'utiliser un tableau d'éléments du
 type \a T de la même manière qu'un tableau C standard. Elle maintient en
 plus la taille du tableau. La fonction
 size() permet de connaître le nombre d'éléments du tableau et l'opérateur
 [] permet d'accéder à un élément donné.

 Cette classe ne gère aucune mémoire et c'est aux classes dérivées
 (ArrayAllocT,...) de gérer la mémoire et de notifier l'instance par setPtr().

 Le contructeur et l'opérateur de recopie ne font que recopier les pointeurs
 sans réallouer la mémoire. Il faut donc les utiliser avec précaution.
 

 En mode débug, les accès par l'intermédiaire des opérateurs [] sont
 vérifiés et le programme se met en pause automatiquement si un débordement de
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
 for( ArrayView<Real>::const_iter i(a); i(); ++i )
 cout << *i;  
 \endcode

 L'exemple suivant fait la somme des 3 premiers éléments du tableau:

 \code
 Real sum = 0.;
 ArrayView<Real>::iterator b = a.begin();     // Itérateur sur le début du tableau
 ArrayView<Real>::iterator e = a.begin() + 3; // Itérateur sur le 4ème élément du tableau
 for( ; b!=e; ++b )
 sum += *i;
 \endcode

*/
template<class T>
class ArrayView
{
 public:
	
  //! Type des éléments du tableau
  typedef T value_type;
  //! Type de l'itérateur sur un élément du tableau
  typedef value_type* iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef const value_type* const_iterator;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

 public:


 public:

  //! Construit un tableau vide.
  ArrayView() : m_size(0), m_ptr(0) {}
  //! Construit un tableau de dimension \a size
  explicit ArrayView(Integer s,T* ptr) : m_size(s), m_ptr(ptr) {}
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
    ARCANE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& operator[](Integer i) const
  {
    ARCANE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Integer i) const
  {
    ARCANE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Positionne le i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline void setItem(Integer i,const T& v)
  {
    ARCANE_CHECK_AT(i,m_size);
    m_ptr[i] = v;
  }

  //! Retourne la taille du tableau
  inline Integer size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Integer length() const { return m_size; }
  
  //! Retourne un iterateur sur le premier élément du tableau
  inline iterator begin() { return m_ptr; }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  inline iterator end() { return m_ptr+m_size; }
  //! Retourne un iterateur constant sur le premier élément du tableau
  inline const_iterator begin() const { return m_ptr; }
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  inline const_iterator end() const { return m_ptr+m_size; }

  //! Addresse du index-ème élément
  inline T* ptrAt(Integer index)
  {
    ARCANE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Integer index) const
  {
    ARCANE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  // Elément d'indice \a i. Vérifie toujours les débordements
  const T& at(Integer i) const
  {
    arcaneCheckAt(i,m_size);
    return m_ptr[i];
  }

  // Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Integer i,const T& value)
  {
    arcaneCheckAt(i,m_size);
    m_ptr[i] = value;
  }

  //! Remplit le tableau avec la valeur \a o
  inline void fill(T o)
  {
    for( Integer i=0, size=m_size; i<size; ++i )
      m_ptr[i] = o;
  }

  /*!
   * \brief Sous-vue à partir de l'élément \a begin et contenant \a size éléments.
   *
   * Si \a (begin+size) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ArrayView<T> subView(Integer begin,Integer size)
  {
    if (begin>=m_size)
      return ArrayView<T>();
    size = math::min(size,m_size-begin);
    return ArrayView<T>(size,m_ptr+begin);
  }

  /*!
   * \brief Sous-vue constante à partir de l'élément \a begin et contenant \a size éléments.
   *
   * Si \a (begin+size) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Integer begin,Integer size) const
  {
    if (begin>=m_size)
      return ConstArrayView<T>();
    size = math::min(size,m_size-begin);
    return ConstArrayView<T>(size,m_ptr+begin);
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

  /*!\brief Recopie le tableau \a copy_array dans l'instance.
   * Comme aucune allocation mémoire n'est effectuée, le
   * nombre d'éléments de \a copy_array doit être inférieur ou égal au
   * nombre d'éléments courant. S'il est inférieur, les éléments du
   * tableau courant situés à la fin du tableau sont inchangés
   */
  template<class U>
  inline void copy(const U& copy_array)
    {
      ARCANE_ASSERT( (copy_array.size()<=m_size), ("Bad size %d %d",copy_array.size(),m_size) );
      const T* copy_begin = copy_array.begin();
      T* to_ptr = m_ptr;
      Integer size = copy_array.size();
      for( Integer i=0; i<size; ++i )
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
    { m_ptr = v.m_ptr; m_size = v.m_size; }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Cette méthode est dangereuse. Le pointeur retourné peut être
   * invalidé dès que le nombre d'éléments du tableau change.
   */
  inline T* unguardedBasePointer()
  { return m_ptr; }

  /*!
   * \brief Pointeur sur la mémoire allouée.
   *
   * \warning Cette méthode est dangereuse. Le pointeur retourné peut être
   * invalidé dès que le nombre d'éléments du tableau change.
   */
  inline const T* unguardedBasePointer() const
  { return m_ptr; }

 protected:
	
  /*!
   * \brief Retourne un pointeur sur le tableau
   *
   * \warning Il est préférable de ne pas utiliser cette méthode pour
   * accéder à un élément du tableau car
   * ce pointeur peut être invalidé par un redimensionnement du tableau.
   * De plus, accéder aux éléments du tableau par ce pointeur ne permet
   * aucune vérification de dépassement, même en mode DEBUG.
   */
  inline T* _ptr() { return m_ptr; }
  /*!
   * \brief Retourne un pointeur sur le tableau
   *
   * \warning Il est préférable de ne pas utiliser cette méthode pour
   * accéder à un élément du tableau car
   * ce pointeur peut être invalidé par un redimensionnement du tableau.
   * De plus, accéder aux éléments du tableau par ce pointeur ne permet
   * aucune vérification de dépassement, même en mode DEBUG.
   */
  inline const T* _ptr() const { return m_ptr; }
  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setArray(T* v,Integer s){ m_ptr = v; m_size = s; }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setPtr(T* v)
    { m_ptr = v; }

  /*!
   * \brief Modifie la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setSize(Integer s)
    { m_size = s; }

 private:

  Integer m_size; //!< Nombre d'éléments du tableau
  T* m_ptr;  //!< Pointeur sur le tableau
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue constante d'un tableau de type \a T.
 *
 * \ingroup Collection

 Cette classe encapsule un tableau C constant standard (pointeur) et son nombre
 d'éléments. L'accès à ses éléments se fait par l'opérateur operator[]().
 La méthode base() permet d'obtenir le pointeur du tableau pour le passer
 aux fonctions C standard.

 L'instance conserve juste un pointeur sur le début du tableau C et ne fait
 aucune gestion mémoire. Le développeur doit s'assurer que le pointeur
 reste valide tant que l'instance existe. En particulier, la vue est invalidée
 dès que le tableau de référence est modifié via un appel à une fonction non-const.

 Les éléments du tableau ne peuvent pas être modifiés.

 En mode débug, une vérification de débordement est effectuée lors de l'accès
 à l'opérateur operator[]().
 */
template<class T>
class ConstArrayView
{
 private:

 protected:

 public:
	
  //! Type des éléments du tableau
  typedef T value_type;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef const value_type* const_iterator;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

 public:

  //! Construit un tableau vide.
  ConstArrayView() : m_size(0), m_ptr(0) {}
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
  : m_size(from.size()), m_ptr(from.begin())
    {
    }

  /*! \brief Opérateur de recopie.
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
      m_ptr  = from.begin();
      return (*this);
    }
  // (HP) The destructor should be virtual because is super class-- Error if it's done
  // virtual ~ConstArrayView() { }

 public:

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a begin et contenant \a size éléments.
   *
   * Si \a (begin+size) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subView(Integer begin,Integer size) const
  {
    if (begin>=m_size)
      return ConstArrayView<T>();
    size = math::min(size,m_size-begin);
    return ConstArrayView<T>(size,m_ptr+begin);
  }

  /*!
   * \brief Sous-vue (constante) à partir de l'élément \a begin et contenant \a size éléments.
   *
   * Si \a (begin+size) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Integer begin,Integer size) const
  {
    return subView(begin,size);
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
    ARCANE_CHECK_AT(ibegin+isize,n);
    return ConstArrayView<T>(isize,m_ptr+ibegin);
  }

  //! Addresse du index-ème élément
  inline const T* ptrAt(Integer index) const
  {
    ARCANE_CHECK_AT(index,m_size);
    return m_ptr+index;
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T& operator[](Integer i) const
  {
    ARCANE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  inline const T& item(Integer i) const
  {
    ARCANE_CHECK_AT(i,m_size);
    return m_ptr[i];
  }

  //! Nombre d'éléments du tableau
  inline Integer size() const { return m_size; }
  //! Nombre d'éléments du tableau
  inline Integer length() const { return m_size; }
  //! Iterateur sur le premier élément du tableau
  inline const_iterator begin() const { return m_ptr; }
  //! Iterateur sur le premier élément après la fin du tableau
  inline const_iterator end() const { return m_ptr+m_size; }
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
   * \warning Cette méthode est dangereuse. Le pointeur retourné peut être
   * invalidé dès que le nombre d'éléments du tableau change.
   */
  inline const T* unguardedBasePointer() const
  { return m_ptr; }

 protected:

 private:

  Integer m_size; //!< Nombre d'éléments 
  const T* m_ptr; //!< Pointeur sur le début du tableau
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
