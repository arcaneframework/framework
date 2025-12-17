// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Collection.h                                                (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'une collection.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_COLLECTION_H
#define ARCCORE_COMMON_COLLECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/AutoRef2.h"

#include "arccore/common/Event.h"

#include <algorithm>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un objet avec compteur de référence.
 *
 * Ces objets sont gérés par compteur de référence.
 */
class ARCCORE_COMMON_EXPORT ObjectImpl
{
 public:

  ObjectImpl()
  : m_ref_count(0)
  {}
  ObjectImpl(const ObjectImpl& rhs) = delete;
  virtual ~ObjectImpl() {}
  ObjectImpl& operator=(const ObjectImpl& rhs) = delete;

 public:

  //! Incrémente le compteur de référence
  void addRef() { ++m_ref_count; }
  //! Décrémente le compteur de référence
  void removeRef()
  {
    Int32 r = --m_ref_count;
    if (r < 0)
      _noReferenceErrorCallTerminate(this);
    if (r == 0)
      deleteMe();
  }
  //! Retourne la valeur du compteur de référence
  Int32 refCount() const { return m_ref_count.load(); }

 public:

  //! Détruit cet objet
  virtual void deleteMe() { delete this; }

 private:

  std::atomic<Int32> m_ref_count; //!< Nombre de références sur l'objet.

 private:

  static void _noReferenceErrorCallTerminate(const void* ptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EnumeratorImplBase;

extern "C" ARCCORE_COMMON_EXPORT void throwOutOfRangeException();
extern "C" ARCCORE_COMMON_EXPORT void throwNullReference();


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
class ARCCORE_COMMON_EXPORT EnumeratorImplBase
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
class ARCCORE_COMMON_EXPORT EnumeratorBase
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

  AutoRef2<EnumeratorImplBase> m_impl; //!< Implémentation
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
/*!
 * \internal
 * \brief Arguments d'un évènement envoyé par une collection.
 *
 * \ingroup Collection
 *
 * Une collection peut envoyé 4 types d'évènements, indiqué par le champs
 * \a m_action:
 * \arg Clear lorsque tous les éléments de la liste sont supprimés
 * \arg Insert lorsqu'un élément de la liste est ajouté.
 * \arg Remove lorsqu'un élément de la liste est supprimé.
 * \arg Set
 *
 */
class CollectionEventArgs
{
 public:

  enum eAction
  {
    ClearComplete,
    InsertComplete,
    RemoveComplete,
    SetComplete
  };

 public:

  CollectionEventArgs(eAction aaction, void* aobject, Integer aposition)
  : m_action(aaction)
  , m_object(aobject)
  , m_position(aposition)
  {}

 public:

  eAction action() const { return m_action; }
  void* object() const { return m_object; }
  Integer position() const { return m_position; }

 private:

  eAction m_action;
  void* m_object;
  Integer m_position;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Evènements envoyés par une Collection
 * \relates Collection
 */
typedef EventObservable<const CollectionEventArgs&> CollectionChangeEventHandler;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief classe de base d'implémentation d'une collection.
 *
 * Une collection est un objet contenant des éléments (i.e. un conteneur).
 *
 * Il est possible de parcourir les éléments d'une collection au moyen
 * d'un énumerateur obtenu par enumerator(). L'énumérateur ainsi obtenu
 * est générique quel que soit le type de la collection. Il est par
 * conséquent moins performant qu'un énumérateur dédié à un type et il
 * vaut mieux utiliser ce dernier si cela est possible.
 *
 * Une collection génère des événements lorsque des éléments sont supprimés,
 * insérés ou modifiés. Il est possible d'enregistrer un handler pour
 * obtenir ces évènements avec change().
 *
 * Les opérations constantes sont threadsafe.
 *
 * Cette classe est destinée à être dérivée pour chaque implémentation
 * d'une collection.
 *
 * \sa EnumeratorImpl
 */
class CollectionImplBase
: public ObjectImpl
{
 public:

  //! Type indexant le tableau
  typedef Integer size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

 public:

  //! Construit une collection vide
  CollectionImplBase() = default;
  //! Construit une collection avec \a acount éléments
  explicit CollectionImplBase(Integer acount)
  : m_count(acount)
  {}
  /*!\brief Opérateur de recopie.
   * les handlers d'évènements ne sont pas recopiés. */
  CollectionImplBase(const CollectionImplBase& from) = delete;

 public:

  //! Retourne le nombre d'éléments de la collection
  Integer count() const { return m_count; }
  //! Supprime tous les éléments de la collection
  virtual void clear() = 0;

 public:

  //! Evènement envoyé avant de supprimer tous les éléments
  virtual void onClear() {}
  //! Evènement envoyé lorsque tous les éléments ont été supprimés
  virtual void onClearComplete()
  {
    _sendEvent(CollectionEventArgs::ClearComplete, 0, 0);
  }
  //! Evènement envoyé avant d'insérer un élément
  virtual void onInsert() {}
  //! Evènement envoyé après insertion d'un élément
  virtual void onInsertComplete(void* object, Integer position)
  {
    _sendEvent(CollectionEventArgs::InsertComplete, object, position);
  }
  //! Evènement envoyé avant de supprimer un élément
  virtual void onRemove() {}
  //! Evènement envoyé après supression d'un élément
  virtual void onRemoveComplete(void* object, Integer position)
  {
    _sendEvent(CollectionEventArgs::RemoveComplete, object, position);
  }
  virtual void onSet() {}
  virtual void onSetComplete(void* object, Integer position)
  {
    _sendEvent(CollectionEventArgs::SetComplete, object, position);
  }
  virtual void onValidate() {}

 public:

  //! Retourne un énumérateur générique sur la collection.
  virtual EnumeratorImplBase* enumerator() const = 0;

 public:

  CollectionChangeEventHandler& change() { return m_collection_handlers; }

 protected:

  void _setCount(Integer acount) { m_count = acount; }

 private:

  Integer m_count = 0;
  CollectionChangeEventHandler m_collection_handlers;

 private:

  void _sendEvent(CollectionEventArgs::eAction action, void* object, Integer position)
  {
    CollectionEventArgs args(action, object, position);
    m_collection_handlers.notify(args);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief classe de base d'implémentation d'une collection typée.
 */
template <class T>
class CollectionImplT
: public CollectionImplBase
{
 public:

  typedef const T& ObjectRef;
  typedef T* ObjectIterator;
  typedef const T* ConstObjectIterator;

 public:

  CollectionImplT()
  : CollectionImplBase()
  {}
  virtual ~CollectionImplT() {}

 public:

  virtual ObjectIterator begin() = 0;
  virtual const T* begin() const = 0;
  virtual ObjectIterator end() = 0;
  virtual const T* end() const = 0;

  virtual T* begin2() const = 0;
  virtual T* end2() const = 0;

 public:

  //! Applique le fonctor \a f à tous les éléments de la collection
  template <class Function> Function
  each(Function f)
  {
    std::for_each(begin(), end(), f);
    return f;
  }

 public:

  virtual void add(ObjectRef value) = 0;
  virtual bool remove(ObjectRef value) = 0;
  virtual void removeAt(Integer index) = 0;
  virtual bool contains(ObjectRef value) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une collection.
 * \ingroup Collection
 */
class ARCCORE_COMMON_EXPORT CollectionBase
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
#ifdef ARCCORE_CHECK
    ARCCORE_CHECK_POINTER(m_ref.get());
#endif
    return m_ref.get();
  }
  const Impl* _noNullRef() const
  {
#ifdef ARCCORE_CHECK
    ARCCORE_CHECK_POINTER(m_ref.get());
#endif
    return m_ref.get();
  }

  void _setRef(Impl* new_impl)
  {
    m_ref = new_impl;
  }

 private:

  AutoRef2<Impl> m_ref;
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
