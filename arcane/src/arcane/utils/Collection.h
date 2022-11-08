// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Collection.h                                                (C) 2000-2022 */
/*                                                                           */
/* Classe de base d'une collection.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_COLLECTION_H
#define ARCANE_UTILS_COLLECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AutoRef.h"
#include "arcane/utils/CollectionImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une collection.
 * \ingroup Collection
 */
class ARCANE_UTILS_EXPORT CollectionBase
{
 private:

  typedef CollectionImplBase Impl;

 public:

  CollectionBase(const CollectionBase& rhs)
  : m_ref(rhs.m_ref)
  {}
  ~CollectionBase() {}

 public:

  /*! \brief Créé une collection nulle.
   *
   * L'instance n'est pas utilisable tant qu'elle n'a pas été affectée
   * à une collection non nulle (par exemple un vecteur).
   */
  CollectionBase() = default;
  CollectionBase& operator=(const CollectionBase& rhs)
  {
    m_ref = rhs.m_ref;
    return *this;
  }

 protected:

  explicit CollectionBase(Impl* vb)
  : m_ref(vb)
  {}

 public:

  //! Supprime tous les éléments de la collection
  void clear() { m_ref->clear(); }
  //! Nombre d'éléments de la collection
  Integer count() const { return m_ref->count(); }
  //! True si la collection est vide
  bool empty() const { return count() == 0; }
  //! Evènement invoqués lorsque la collection change
  CollectionChangeEventHandler& change() { return m_ref->change(); }

 protected:

  Impl* _ref() { return m_ref.get(); }
  const Impl* _ref() const { return m_ref.get(); }

  Impl* _noNullRef()
  {
#ifdef ARCANE_CHECK
    Arcane::arcaneCheckNull(m_ref.get());
#endif
    return m_ref.get();
  }
  const Impl* _noNullRef() const
  {
#ifdef ARCANE_CHECK
    Arcane::arcaneCheckNull(m_ref.get());
#endif
    return m_ref.get();
  }

  void _setRef(Impl* new_impl)
  {
    m_ref = new_impl;
  }

 private:

  AutoRefT<Impl> m_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une collection fortement typée.
 * \ingroup Collection
 */
template <typename T>
class Collection
: public CollectionBase
{
 private:

  typedef CollectionImplT<T> Impl;

 public:

  typedef const T& ObjectRef;
  typedef T& Ref;
  typedef T* Iterator;

 public:

  //! Type d'un itérateur sur toute la collection
  typedef EnumeratorT<T> Enumerator;

 public:

  /*!
   * \brief Créé une collection nulle.
   *
   * L'instance n'est pas utilisable tant qu'elle n'a pas été affectée
   * à une collection non nulle.
   */
  Collection() = default;

 protected:

  explicit Collection(Impl* vb)
  : CollectionBase(vb)
  {}

 public:

  Enumerator enumerator() const
  {
    return Enumerator(_cast().enumerator());
  }

  Iterator begin() { return _cast().begin(); }
  Iterator end() { return _cast().end(); }
  Ref front() { return *begin(); }

 public:

  bool remove(ObjectRef value) { return _cast().remove(value); }
  void removeAt(Integer index) { return _cast().removeAt(index); }
  void add(ObjectRef value) { _cast().add(value); }
  bool contains(ObjectRef value) const { return _cast().contains(value); }

 public:

  //! Applique le fonctor \a f à tous les éléments de la collection
  template <class Function> Function
  each(Function f) { return _cast().each(f); }

 private:

  Impl& _cast() { return *static_cast<Impl*>(_noNullRef()); }
  const Impl& _cast() const { return *static_cast<const Impl*>(_ref()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
