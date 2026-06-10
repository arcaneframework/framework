// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReferenceCounterImpl.h                                      (C) 2000-2026 */
/*                                                                           */
/* Implementations related to the reference counter manager.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFERENCECOUNTERIMPL_H
#define ARCCORE_BASE_REFERENCECOUNTERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "ArccoreGlobal.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/RefBase.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file ReferenceCounterImpl.h
 *
 * This file should only be included by implementation classes
 * using a reference counter. For declarations, use
 * the file 'RefDeclarations.h'
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> ARCCORE_EXPORT void
ExternalReferenceCounterAccessor<T>::
addReference(T* t)
{
  if constexpr (requires { t->_internalAddReference(); })
    t->_internalAddReference();
  else
    t->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T> ARCCORE_EXPORT void
ExternalReferenceCounterAccessor<T>::
removeReference(T* t)
{
  if constexpr (requires { t->_internalRemoveReference(); }) {
    bool need_destroy = t->_internalRemoveReference();
    if (need_destroy)
      delete t;
  }
  else
    t->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Thread-safe implementation of a reference counter.
 *
 * The implementation uses a std::atomic to maintain the number of
 * references.
 *
 * The removeReference() method destroys the instance when this counter
 * of references reaches 0.
 *
 * This class is internal to Arcane.
 */
class ARCCORE_BASE_EXPORT ReferenceCounterImpl
{
  template <typename InstanceType> friend class impl::ReferenceCounterWrapper;

 public:

  virtual ~ReferenceCounterImpl() = default;

 public:

  ARCCORE_DEPRECATED_REASON("Y2025: use _internalAddReference() instead")
  void addReference()
  {
    ++m_nb_ref;
  }

  ARCCORE_DEPRECATED_REASON("Y2025: use _internalRemoveReference() instead")
  void removeReference()
  {
    // Decrements and returns the previous value.
    // If it equals 1, it means there are no more references
    // on the object and it must be destroyed.
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1) {
      if (_destroyThisReference())
        delete this;
    }
  }

 public:

  void _internalAddReference()
  {
    ++m_nb_ref;
  }
  bool _internalRemoveReference()
  {
    // Decrements and returns the previous value.
    // If it equals 1, it means there are no more references
    // on the object and it must potentially be destroyed.
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      return _destroyThisReference();
    return false;
  }

 private:

  // Methods accessible only from ReferenceCounterWrapper
  void _setExternalDeleter(RefBase::DeleterBase* v)
  {
    delete m_external_deleter;
    m_external_deleter = v;
  }
  RefBase::DeleterBase* _externalDeleter() const
  {
    return m_external_deleter;
  }

 private:

  std::atomic<Int32> m_nb_ref = 0;
  RefBase::DeleterBase* m_external_deleter = nullptr;

 private:

  //! Returns true if the instance must be destroyed by calling operator delete()
  bool _destroyThisReference()
  {
    if (!m_external_deleter)
      return true;
    bool do_delete = false;
    if (!m_external_deleter->m_no_destroy) {
      bool is_destroyed = m_external_deleter->_destroyHandle(this, m_external_deleter->m_handle);
      if (!is_destroyed) {
        do_delete = true;
      }
    }
    delete m_external_deleter;
    m_external_deleter = nullptr;
    return do_delete;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * Generic macro to define methods managing reference counters.
 */
#define ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS(OPTIONAL_OVERRIDE) \
 private: \
\
  using BaseCounterType = ::Arcane::ReferenceCounterImpl; \
\
 public: \
\
  BaseCounterType* _internalReferenceCounter() OPTIONAL_OVERRIDE \
  { \
    return this; \
  } \
  void _internalAddReference() OPTIONAL_OVERRIDE \
  { \
    BaseCounterType::_internalAddReference(); \
  } \
  bool _internalRemoveReference() OPTIONAL_OVERRIDE \
  { \
    return BaseCounterType::_internalRemoveReference(); \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to define methods managing counters
 * of references.
 *
 * This macro is used in a class implementing an interface
 * for which the macro ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS()
 * was used. The implementation class must inherit from
 * ReferenceCounterImpl. For example:
 *
 * \code
 * class MyClass
 * : public ReferenceCounterImpl
 * , public IMyInterface
 * {
 *   ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
 *  public:
 *   void myMethod1() override;
 * };
 * \endcode
 */
#define ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS() \
  ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS(override)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to define methods and types for a class that
 * uses a reference counter.
 *
 * This macro must be used for a class for which
 * the macro ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS() was used. It must
 * be located in a single translation unit (a '.cc' file, for
 * example) and be used in the Arccore namespace. For example:
 *
 * \code
 * namespace Arccore
 * {
 *   ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(MyNamespace::MyClass);
 * }
 * \endcode
 */
#define ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(class_name) \
  template class ExternalReferenceCounterAccessor<class_name>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
