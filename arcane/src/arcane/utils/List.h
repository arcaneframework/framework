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
#ifndef ARCANE_UTILS_LIST_H
#define ARCANE_UTILS_LIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ListImpl.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Ptr.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur générique pour un tableau
 */
class ARCANE_UTILS_EXPORT ListEnumeratorBase
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
class ARCANE_UTILS_EXPORT ListConstEnumeratorBase
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
