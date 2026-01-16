// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* List.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Classe collection tableau.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_LIST_H
#define ARCCORE_COMMON_LIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Collection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCCORE_COMMON_EXPORT void throwOutOfRangeException();
extern "C" ARCCORE_COMMON_EXPORT void throwNullReference();

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
/*!
 * \brief Enumérateur générique pour un tableau
 */
class ARCCORE_COMMON_EXPORT ListEnumeratorBase
{
 public:

  typedef void* VoidPtr;

 public:

  ListEnumeratorBase(VoidPtr begin, VoidPtr end, Integer elem_size)
  : m_begin(begin)
  , m_end(end)
  , m_current(begin)
  , m_elem_size(elem_size)
  {}

 public:

  void reset() { m_current = m_begin; }
  bool moveNext()
  {
    m_current = (char*)m_current + m_elem_size;
    return m_current < m_end;
  }
  VoidPtr current() { return m_current; }

  bool operator++() { return moveNext(); }
  VoidPtr operator*() { return current(); }

 protected:

  VoidPtr m_begin;
  VoidPtr m_end;
  VoidPtr m_current;
  Integer m_elem_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur générique constant pour un tableau
 */
class ARCCORE_COMMON_EXPORT ListConstEnumeratorBase
{
 public:

  typedef const void* VoidPtr;

 public:

  ListConstEnumeratorBase(VoidPtr begin, VoidPtr end, Integer elem_size)
  : m_begin(begin)
  , m_end(end)
  , m_current(begin)
  , m_elem_size(elem_size)
  {}

 public:

  void reset() { m_current = m_begin; }
  bool moveNext()
  {
    m_current = (const char*)m_current + m_elem_size;
    return m_current < m_end;
  }
  VoidPtr current() { return m_current; }

  bool operator++() { return moveNext(); }
  VoidPtr operator*() { return current(); }

 protected:

  VoidPtr m_begin;
  VoidPtr m_end;
  VoidPtr m_current;
  Integer m_elem_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> class List;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur typé pour un tableau
 */
template <typename T>
class ListEnumeratorT
: public ListEnumeratorBase
{
 public:

  typedef T* Ptr;
  typedef T& Ref;

 public:

  ListEnumeratorT(Ptr begin, Ptr end)
  : ListEnumeratorBase(begin - 1, end, sizeof(T))
  {}
  ListEnumeratorT(const List<T>& collection);

 public:

  inline bool moveNext()
  {
    Ptr v = reinterpret_cast<Ptr>(m_current);
    ++v;
    m_current = v;
    return m_current < m_end;
  }
  inline Ptr current()
  {
    return reinterpret_cast<Ptr>(m_current);
  }
  inline bool operator++()
  {
    return moveNext();
  }
  inline Ref operator*()
  {
    return *current();
  }
  inline Ptr operator->()
  {
    return current();
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ListConstEnumeratorT
: public ListConstEnumeratorBase
{
 public:

  typedef const T* Ptr;
  typedef const T& Ref;

 public:

  ListConstEnumeratorT(Ptr begin, Ptr end)
  : ListConstEnumeratorBase(begin - 1, end, sizeof(T))
  {}
  ListConstEnumeratorT(const List<T>& collection);

 public:

  inline bool moveNext()
  {
    Ptr v = reinterpret_cast<Ptr>(m_current);
    ++v;
    m_current = v;
    return m_current < m_end;
  }
  inline Ptr current()
  {
    return reinterpret_cast<Ptr>(m_current);
  }
  inline bool operator++()
  {
    return moveNext();
  }
  inline Ref operator*()
  {
    return *current();
  }
  inline Ptr operator->()
  {
    return current();
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'une collection d'éléments sous forme de vecteur.
 */
template <typename T>
class List
: public Collection<T>
{
 public:

  typedef Collection<T> BaseClass;

 private:

  typedef ListImplT<T> Impl;

 public:

  typedef Impl Vec;
  typedef typename Vec::value_type value_type;
  typedef typename Vec::iterator iterator;
  typedef typename Vec::const_iterator const_iterator;
  typedef typename Vec::pointer pointer;
  typedef typename Vec::const_pointer const_pointer;
  typedef typename Vec::reference reference;
  typedef typename Vec::const_reference const_reference;

  //! Type d'un itérateur constant sur tout le tableau
  typedef ListEnumeratorT<T> Enumerator;

  typedef ConstIterT<List<T>> const_iter;

 public:

  List()
  : BaseClass(new Impl())
  {}
  List(const ConstArrayView<T>& from)
  : BaseClass(new Impl(from))
  {}
  List(const ArrayView<T>& from)
  : BaseClass(new Impl(from))
  {}
  explicit List(const EnumeratorT<T>& from)
  : BaseClass(new Impl(from))
  {}

 public:

  void resize(Integer new_size)
  {
    _cast().resize(new_size);
  }

  T& operator[](Integer i)
  {
    return _cast()[i];
  }

  const T& operator[](Integer i) const
  {
    return _cast()[i];
  }

  //! Clone la collection \a base
  void clone(const Collection<T>& base)
  {
    //Impl* n = new Impl(base);
    //BaseClass::_setRef(n);
    _cast().assign(base);
  }

  //! Clone la collection
  List<T> clone() const
  {
    return List<T>(new Impl(*this));
  }

 public:

  const_iterator begin() const { return _cast().begin(); }
  const_iterator end() const { return _cast().end(); }
  iterator begin() { return _cast().begin(); }
  iterator end() { return _cast().end(); }

  T* begin2() const { return _cast().begin2(); }
  T* end2() const { return _cast().end2(); }

  //! Applique le fonctor \a f à tous les éléments du tableau
  template <typename Function> Function
  each(Function f)
  {
    return _cast().each(f);
  }

  template <typename Function> iterator
  find_if(Function f)
  {
    iterator _end = end2();
    iterator i = std::find_if(begin2(), _end, f);
    if (i != _end)
      return i;
    return _end;
  }

  template <typename Function> const_iterator
  find_if(Function f) const
  {
    const_iterator _end = end();
    const_iterator i = std::find_if(begin(), _end, f);
    if (i != _end)
      return i;
    return _end;
  }

 protected:

  List(Impl* from)
  : BaseClass(from)
  {}

 private:

  Impl& _cast() { return *static_cast<Impl*>(this->_noNullRef()); }
  const Impl& _cast() const { return *static_cast<const Impl*>(this->_noNullRef()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline ListEnumeratorT<T>::
ListEnumeratorT(const List<T>& collection)
: ListEnumeratorBase(collection.begin2() - 1, collection.end2(), sizeof(T))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline ListConstEnumeratorT<T>::
ListConstEnumeratorT(const List<T>& collection)
: ListConstEnumeratorBase(collection.begin2() - 1, collection.end2(), sizeof(T))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
