// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Enumerator.h                                                (C) 2000-2022 */
/*                                                                           */
/* Enumérateurs.                                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ENUMERATOR_H
#define ARCANE_ENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ObjectImpl.h"
#include "arcane/utils/AutoRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un énumérateur.
 *
 * Cette classe sert de classe de base à toutes les implémentations
 * d'itérateurs.
 * Cette classe ne doit pas être utilisée directement: pour effectuer une
 * énumération, il faut utiliser la classe Enumerator ou une de ces
 * classes dérivée.
 *
 * \sa Enumerator
 */
class ARCANE_UTILS_EXPORT EnumeratorImplBase
: public ObjectImpl
{
 public:

  /*! \brief Remet à zéro l'énumérateur.
  *
  * Positionne l'énumérateur juste avant le premier élément de la collection.
   * Il faut faire un moveNext() pour le rendre valide.
   */
  virtual void reset() = 0;
  /*! \brief Avance l'énumérateur sur l'élément suivant de la collection.
   *
   * \retval true si l'énumérateur n'a pas dépassé le dernier élément. Dans
   * ce cas l'appel à current() est valide.
   * \retval false si l'énumérateur a dépassé le derniere élément. Dans ce
   * cas tout appel suivant à cette méthode retourne \a false et l'appel
   * à current() n'est pas valide.
   */
  virtual bool moveNext() = 0;
  //! Objet courant de l'énumérateur.
  virtual void* current() = 0;
  //! Objet courant de l'énumérateur.
  virtual const void* current() const = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Enumérateur générique.
 *
 * Cette classe permet d'itérer de manière générique sur une collection,
 * sans connaître le type des éléments de la collection. Pour une itération
 * utilisant un typage fort, il faut utiliser la classe template EnumeratorT.
 *
 * Exemple d'utilisation d'un énumérateur:
 *
 * \code
 * VectorT<int> integers;
 * for( Enumerator i(integers.enumerator()); ++i; )
 *   cout << i.current() << '\n';
 * \endcode
 */
class ARCANE_UTILS_EXPORT EnumeratorBase
{
 public:

  //! Contruit un énumérateur nul.
  EnumeratorBase() = default;

  /*!
   * \brief Contruit un énumérateur associé à l'implémentation \a impl.
   *
   * L'instance devient propriétaire de l'implémentation qui est détruite
   * lorsque l'instance est détruite.
   */
  explicit EnumeratorBase(EnumeratorImplBase* impl)
  : m_impl(impl)
  {}

 public:

  void reset() { m_impl->reset(); }
  bool moveNext() { return m_impl->moveNext(); }
  void* current() { return m_impl->current(); }
  const void* current() const { return m_impl->current(); }

 public:

  //! Avance l'énumérateur sur l'élément suivant.
  bool operator++() { return moveNext(); }

 protected:

  EnumeratorImplBase* _impl() { return m_impl.get(); }
  const EnumeratorImplBase* _impl() const { return m_impl.get(); }

 private:

  AutoRefT<EnumeratorImplBase> m_impl; //!< Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Enumérateur typé.
 *
 * \todo utiliser des traits pour les types références, pointeur et valeur
 */
template <class T>
class EnumeratorT
: public EnumeratorBase
{
 public:

  EnumeratorT() = default;
  EnumeratorT(const Collection<T>& collection);
  explicit EnumeratorT(EnumeratorImplBase* impl)
  : EnumeratorBase(impl)
  {}

 public:

  const T& current() const { return *_currentPtr(); }
  T& current() { return *_currentPtr(); }

 public:

  const T& operator*() const { return current(); }
  T& operator*() { return current(); }
  const T* operator->() const { return _currentPtr(); }
  T* operator->() { return _currentPtr(); }

 private:

  T* _currentPtr()
  {
    return reinterpret_cast<T*>(_impl()->current());
  }
  const T* _currentPtr() const
  {
    return reinterpret_cast<const T*>(_impl()->current());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> inline EnumeratorT<T>::
EnumeratorT(const Collection<T>& collection)
: EnumeratorBase(collection.enumerator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
