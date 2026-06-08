// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadPrivate.h                                             (C) 2000-2025 */
/*                                                                           */
/* Class for storing a specific value per thread.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_THREADPRIVATE_H
#define ARCCORE_CONCURRENCY_THREADPRIVATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include "arccore/concurrency/GlibAdapter.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Container for thread-private values.
 *
 * initialize() must be called before using the setValue()/getValue() methods.
 * This initialize() method can be called multiple times.
 *
 * \deprecated Use 'thread_local' from C++11.
 */
class ARCCORE_CONCURRENCY_EXPORT ThreadPrivateStorage
{
 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivateStorage();
  ~ThreadPrivateStorage();

 public:

  /*!
   * \brief Initializes the key containing thread-private values.
   * This method can be called multiple times and does nothing if
   * the key has already been initialized.
   *
   * \warning This method is not thread-safe. The user must therefore
   * be careful during the first call.
   */
  void initialize();
  void* getValue();
  void setValue(void* v);

 private:

  GlibPrivate* m_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class allowing an instance of an object to be stored per thread.
 *
 * \deprecated Use 'thread_local' from C++11.
 */
class ARCCORE_CONCURRENCY_EXPORT ThreadPrivateBase
{
 public:

  class ICreateFunctor
  {
   public:

    virtual ~ICreateFunctor() {}
    virtual void* createInstance() = 0;
  };

 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivateBase(ThreadPrivateStorage* key, ICreateFunctor* create_functor)
  : m_key(key)
  , m_create_functor(create_functor)
  {
  }

  ~ThreadPrivateBase()
  {
  }

 public:

  /*!
   * \brief Retrieves the instance specific to the current thread.
   *
   * If it does not yet exist, it is created via
   * the functor passed as an argument to the constructor.
   *
   * \warning This method must not be called until
   * the associated key (ThreadPrivateStorage) has been initialized
   * by calling ThreadPrivateStorage::initialize().
   */
  void* item();

 private:

  ThreadPrivateStorage* m_key;
  GlibMutex m_mutex;
  ICreateFunctor* m_create_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing an instance of a type per thread.
 *
 * The container allowing values to be stored must be passed as an argument
 * to the constructor. This container must have been initialized
 * via ThreadPrivateStorage::initialize() before using this class.
 *
 * This class only has one method item()
 * allowing retrieval of an instance of type \a T per
 * thread. On the first call to item() for a given thread,
 * an instance of \a T is constructed.
 * The type \a T must have a default constructor
 * and must have a \a build() method.
 * \threadsafeclass
 *
 * \deprecated Use 'thread_local' from C++11.
 */
template <typename T>
class ThreadPrivate
: private ThreadPrivateBase::ICreateFunctor
{
 public:

  ARCCORE_DEPRECATED_REASON("Y2022; This class is deprecated. Use 'thread_local' specifier.")
  ThreadPrivate(ThreadPrivateStorage* key)
  : m_storage(key, this)
  {
  }

  ~ThreadPrivate()
  {
    for (T* item : m_allocated_items)
      delete item;
  }

 public:

  //! Instance specific to the current thread.
  T* item()
  {
    return (T*)(m_storage.item());
  }

 private:

  void* createInstance() override
  {
    T* new_ptr = new T();
    new_ptr->build();
    m_allocated_items.push_back(new_ptr);
    return new_ptr;
  }

 private:

  std::vector<T*> m_allocated_items;
  ThreadPrivateBase m_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
