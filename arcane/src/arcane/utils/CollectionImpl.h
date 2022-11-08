// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CollectionImpl.h                                            (C) 2000-2022 */
/*                                                                           */
/* Implémentation de la classe de base d'une collection.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_COLLECTIONIMPL_H
#define ARCANE_UTILS_COLLECTIONIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ObjectImpl.h"
#include "arcane/utils/Iostream.h"
// TODO: supprimer ce .h qui n'est pas utilisé.
#include "arcane/utils/Deleter.h"
#include "arcane/utils/Event.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EnumeratorImplBase;

extern "C" ARCANE_UTILS_EXPORT void throwOutOfRangeException();
extern "C" ARCANE_UTILS_EXPORT void throwNullReference();

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
  CollectionImplBase()
  : m_count(0)
  {}
  //! Construit une collection avec \a acount éléments
  CollectionImplBase(Integer acount)
  : m_count(acount)
  {}
  /*!\brief Opérateur de recopie.
   * les handlers d'évènements ne sont pas recopiés. */
  CollectionImplBase(const CollectionImplBase& from)
  : ObjectImpl(from)
  , m_count(from.count())
  {}
  virtual ~CollectionImplBase() {}

 public:

  //! Retourne le nombre d'éléments de la collection
  Integer count() const { return m_count; }
  //! Supprime tous les éléments de la collection
  virtual void clear()
  {
    onClear();
    _setCount(0);
    onClearComplete();
  }

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

  Integer m_count;
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
  virtual ConstObjectIterator find(ObjectRef value) const = 0;
  virtual ObjectIterator find(ObjectRef value) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
