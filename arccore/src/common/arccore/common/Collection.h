// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Collection.h                                                (C) 2000-2025 */
/*                                                                           */
/* Base class for a collection.                                              */
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
 * \brief Base class for an object with a reference counter.
 *
 * These objects are managed by a reference counter.
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

  //! Increments the reference counter
  void addRef() { ++m_ref_count; }
  //! Decrements the reference counter
  void removeRef()
  {
    Int32 r = --m_ref_count;
    if (r < 0)
      _noReferenceErrorCallTerminate(this);
    if (r == 0)
      deleteMe();
  }
  //! Returns the value of the reference counter
  Int32 refCount() const { return m_ref_count.load(); }

 public:

  //! Destroys this object
  virtual void deleteMe() { delete this; }

 private:

  std::atomic<Int32> m_ref_count; //!< Number of references on the object.

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
 * \brief Enumerator interface.
 *
 * This class serves as the base class for all iterator implementations.
 * This class should not be used directly: to perform an enumeration,
 * you must use the Enumerator class or one of its derived classes.
 *
 * \sa Enumerator
 */
class ARCCORE_COMMON_EXPORT EnumeratorImplBase
: public ObjectImpl
{
 public:

  /*!
   * \brief Resets the enumerator.
   *
   * Positions the enumerator just before the first element of the collection.
   * A moveNext() must be performed to make it valid.
   */
  virtual void reset() = 0;
  /*! \brief Advances the enumerator to the next element in the collection.
   *
   * \retval true if the enumerator has not passed the last element. In
   * this case, the call to current() is valid.
   * \retval false if the enumerator has passed the last element. In this
   * case, any subsequent call to this method returns \a false and the call
   * to current() is not valid.
   */
  virtual bool moveNext() = 0;
  //! Current object of the enumerator.
  virtual void* current() = 0;
  //! Current object of the enumerator.
  virtual const void* current() const = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Generic enumerator.
 *
 * This class allows generic iteration over a collection without knowing
 * the type of the collection elements. For iteration using strong typing,
 * you must use the template class EnumeratorT.
 *
 * Example of enumerator usage:
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

  //! Constructs a null enumerator.
  EnumeratorBase() = default;

  /*!
   * \brief Constructs an enumerator associated with the implementation \a impl.
   *
   * The instance becomes the owner of the implementation, which is destroyed
   * when the instance is destroyed.
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

  //! Advances the enumerator to the next element.
  bool operator++() { return moveNext(); }

 protected:

  EnumeratorImplBase* _impl() { return m_impl.get(); }
  const EnumeratorImplBase* _impl() const { return m_impl.get(); }

 private:

  AutoRef2<EnumeratorImplBase> m_impl; //!< Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Typed enumerator.
 *
 * \todo use traits for reference, pointer, and value types
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
 * \brief Arguments of an event sent by a collection.
 *
 * \ingroup Collection
 *
 * A collection can send 4 types of events, indicated by the field
 * \a m_action:
 * \arg Clear when all elements of the list are deleted
 * \arg Insert when an element is added to the list.
 * \arg Remove when an element is deleted from the list.
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
 * \brief Events sent by a Collection
 * \relates Collection
 */
typedef EventObservable<const CollectionEventArgs&> CollectionChangeEventHandler;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for collection implementation.
 *
 * A collection is an object containing elements (i.e., a container).
 *
 * It is possible to iterate over the elements of a collection using
 * an enumerator obtained via enumerator(). The enumerator obtained this
 * way is generic regardless of the collection type. It is therefore less
 * performant than an enumerator dedicated to a type, and it is better
 * to use the latter if possible.
 *
 * A collection generates events when elements are removed,
 * inserted, or modified. It is possible to register a handler to
 * receive these events using change().
 *
 * Constant operations are threadsafe.
 *
 * This class is intended to be derived for each implementation
 * of a collection.
 *
 * \sa EnumeratorImpl
 */
class CollectionImplBase
: public ObjectImpl
{
 public:

  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between array iterator elements
  typedef ptrdiff_t difference_type;

 public:

  //! Constructs an empty collection
  CollectionImplBase() = default;
  //! Constructs a collection with \a acount elements
  explicit CollectionImplBase(Integer acount)
  : m_count(acount)
  {}
  /*!\brief Copy constructor.
   * event handlers are not copied. */
  CollectionImplBase(const CollectionImplBase& from) = delete;

 public:

  //! Returns the number of elements in the collection
  Integer count() const { return m_count; }
  //! Removes all elements from the collection
  virtual void clear() = 0;

 public:

  //! Event sent before removing all elements
  virtual void onClear() {}
  //! Event sent when all elements have been removed
  virtual void onClearComplete()
  {
    _sendEvent(CollectionEventArgs::ClearComplete, 0, 0);
  }
  //! Event sent before inserting an element
  virtual void onInsert() {}
  //! Event sent after inserting an element
  virtual void onInsertComplete(void* object, Integer position)
  {
    _sendEvent(CollectionEventArgs::InsertComplete, object, position);
  }
  //! Event sent before removing an element
  virtual void onRemove() {}
  //! Event sent after removing an element
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

  //! Returns a generic enumerator for the collection.
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
 * \brief base class for implementation of a typed collection.
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

  //! Applies the functor \a f to all elements of the collection
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
 * \brief Base class for a collection.
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

  /*!
   * \brief Creates a null collection.
   *
   * The instance is not usable until it has been assigned
   * to a non-null collection (e.g., a vector).
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

  //! Removes all elements from the collection
  void clear() { m_ref->clear(); }
  //! Number of elements in the collection
  Integer count() const { return m_ref->count(); }
  //! True if the collection is empty
  bool empty() const { return count() == 0; }
  //! Event invoked when the collection changes
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
 * \brief Base class for a strongly typed collection.
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

  //! Type of an iterator over the entire collection
  typedef EnumeratorT<T> Enumerator;

 public:

  /*!
   * \brief Creates a null collection.
   *
   * The instance is not usable until it has been assigned
   * to a non-null collection.
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

  //! Applies the functor \a f to all elements of the collection
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
