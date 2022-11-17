// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ListImpl.h                                                  (C) 2000-2022 */
/*                                                                           */
/* Implémentation de la classe Tableau.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LISTIMPL_H
#define ARCANE_UTILS_LISTIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CollectionImpl.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_UTILS_EXPORT void throwOutOfRangeException();
extern "C" ARCANE_UTILS_EXPORT void throwNullReference();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableau avec allocateur virtuel.
 *
 * C'est à la classe virtuelle de détruire les objets dans le
 * destructeur virtuel.
 */
template <class T>
class ListImplBase
: public CollectionImplT<T>
{
 public:

  typedef CollectionImplT<T> BaseClass;

 public:

  typedef const T& ObjectRef;

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

  //! Type d'un itérateur sur tout le tableau
  typedef IterT<ListImplBase<T>> iter;
  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT<ListImplBase<T>> const_iter;

 public:

  //! Construit un tableau vide.
  ListImplBase() = default;

 public:

  //! Recopie le tableau \a s.
  void assign(const ListImplBase<T>& s)
  {
    _arrayCopy(s);
  }
  //! Recopie le tableau \a s.
  void assign(const ConstArrayView<T>& s)
  {
    _arrayCopy(s);
  }
  //! Recopie le tableau \a s.
  void assign(const ArrayView<T>& s)
  {
    _arrayCopy(s);
  }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  T& operator[](Integer i)
  {
    return m_array[i];
  }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T& operator[](Integer i) const
  {
    return m_array[i];
  }

  //! Retourne un iterateur sur le premier élément du tableau
  iterator begin() override
  {
    return m_array.data();
  }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  iterator end() override
  {
    return m_array.data() + this->count();
  }
  //! Retourne un iterateur constant sur le premier élément du tableau
  const_iterator begin() const override
  {
    return m_array.data();
  }
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  const_iterator end() const override
  {
    return m_array.data() + this->count();
  }

  //! Retourne un iterateur sur le premier élément du tableau
  T* begin2() const override
  {
    return const_cast<T*>(m_array.data());
  }

  //! Retourne un iterateur sur le premier élément après la fin du tableau
  T* end2() const override
  {
    return begin2() + this->count();
  }

 public:

  //! Applique le fonctor \a f à tous les éléments du tableau
  template <class Function> Function
  each(Function f)
  {
    std::for_each(begin(), end(), f);
    return f;
  }

 public:

  /*! \brief Signale qu'il faut réserver de la mémoire pour \a new_capacity éléments
   * Il s'agit juste d'une indication. La classe dérivée est libre de ne
   * pas en tenir compte.
   */
  void reserve(Integer new_capacity)
  {
    m_array.reserve(new_capacity);
  }

  /*!
   * \brief Retourne le nombre d'éléments alloués du tableau.
   *
   * Il s'agit juste d'une indication. La classe dérivée est libre de ne
   * pas en tenir compte.
   */
  Integer capacity() const
  {
    return m_array.capacity();
  }

  void clear() override
  {
    this->onClear();
    m_array.clear();
    this->_setCount(0);
    this->onClearComplete();
  }

  //! Ajoute l'élément \a elem à la fin du tableau
  void add(ObjectRef elem) override
  {
    this->onInsert();
    Integer s = this->count();
    m_array.add(elem);
    this->_setCount(s + 1);
    this->onInsertComplete(_ptr() + s, s);
  }

  bool remove(ObjectRef element) override
  {
    Integer i = 0;
    Integer s = this->count();
    for (; i < s; ++i)
      if (m_array[i] == element) {
        _removeAt(i);
        return true;
      }
    throwOutOfRangeException();
    return false;
  }

  void removeAt(Integer index) override
  {
    Integer s = this->count();
    if (index >= s)
      throwOutOfRangeException();
    _removeAt(index);
  }

  const_iterator find(ObjectRef element) const
  {
    Integer s = this->count();
    for (Int32 i = 0; i < s; ++i)
      if (m_array[i] == element)
        return begin() + i;
    return end();
  }

  bool contains(ObjectRef element) const override
  {
    const_iterator i = find(element);
    return (i != end());
  }

  EnumeratorImplBase* enumerator() const override;

 protected:

  void _removeAt(Integer index)
  {
    Integer s = this->count();
    T* ptr = _ptr();
    T* remove_ob = ptr + index;
    this->onRemove();
    m_array.remove(index);
    this->_setCount(s - 1);
    // TODO: supprimer utilisation de 'remove_ob' car ce n'est pas le bon objet
    this->onRemoveComplete(remove_ob, index);
  }

 public:

  /*!\brief Modifie la taille du tableau.
   * \a new_size est le nouveau nombre d'éléments du tableau.
   */
  void resize(Integer new_size)
  {
    m_array.resize(new_size);
    this->_setCount(new_size);
  }

 protected:

  Integer _capacity() const
  {
    return m_array.capacity();
  }

 protected:

  void _arrayCopy(const ListImplBase<T>& array)
  {
    _arrayCopy(array.begin2(), array.count());
  }
  void _arrayCopy(const ConstArrayView<T>& array)
  {
    _arrayCopy(array.data(), array.size());
  }
  void _arrayCopy(const ArrayView<T>& array)
  {
    _arrayCopy(array.data(), array.size());
  }
  void _arrayCopy(const T* from_ptr, Integer from_size)
  {
    if (from_size == 0) {
      clear();
      return;
    }
    m_array.copy(ConstArrayView<T>(from_size, from_ptr));
    this->_setCount(from_size);
  }

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
  inline T* _ptr()
  {
    return m_array.data();
  }

  inline const T* _ptr() const
  {
    return m_array.data();
  }

 private:

  UniqueArray<T> m_array;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 */
template <class T>
class ListImplT
: public ListImplBase<T>
{
 public:

  typedef ListImplBase<T> BaseClass;

 public:

  ListImplT() {}
  explicit ListImplT(const ConstArrayView<T>& array)
  {
    this->_arrayCopy(array);
  }
  explicit ListImplT(const ArrayView<T>& array)
  {
    this->_arrayCopy(array);
  }
  ListImplT(const ListImplT<T>& array)
  : BaseClass()
  {
    this->_arrayCopy(array);
  }
  explicit ListImplT(const Collection<T>& array)
  : BaseClass()
  {
    for (typename Collection<T>::Enumerator i(array); ++i;) {
      BaseClass::add(*i);
    }
  }
  explicit ListImplT(const EnumeratorT<T>& enumerator)
  : BaseClass()
  {
    for (EnumeratorT<T> i(enumerator); ++i;) {
      BaseClass::add(*i);
    }
  }

  void assign(const Collection<T>& array)
  {
    this->clear();
    for (typename Collection<T>::Enumerator i(array); ++i;) {
      this->add(*i);
    }
  }
  void assign(const EnumeratorT<T>& enumerator)
  {
    this->clear();
    for (EnumeratorT<T> i(enumerator); ++i;) {
      this->add(*i);
    }
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 */
template <class T>
class ListEnumeratorImplT
: public EnumeratorImplBase
{
 public:

  typedef T* Ptr;

 public:

  ListEnumeratorImplT(Ptr begin, Ptr end)
  : m_begin(begin)
  , m_current(begin - 1)
  , m_end(end)
  {}

 public:

  void reset() override { m_current = m_begin - 1; }
  bool moveNext() override
  {
    ++m_current;
    return m_current < m_end;
  }
  void* current() override { return m_current; }
  const void* current() const override { return m_current; }

 private:

  Ptr m_begin;
  Ptr m_current;
  Ptr m_end;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> EnumeratorImplBase* ListImplBase<T>::
enumerator() const
{
  return new ListEnumeratorImplT<T>(begin2(), end2());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
