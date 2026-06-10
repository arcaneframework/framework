// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RefDeclarations.h                                           (C) 2000-2026 */
/*                                                                           */
/* Declarations related to reference management on an instance.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFDECLARATIONS_H
#define ARCCORE_BASE_REFDECLARATIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <type_traits>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file RefDeclarations.h
 *
 * This file contains the declarations and macros for managing classes
 * using reference counters. For implementation, you must use
 * the file 'ReferenceCounterImpl.h'
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The ExternalReferenceCounterAccessor class must remain in the
// Arccore namespace for compatibility with existing code and the macro
// ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS.
namespace Arccore
{
template <class T>
class ExternalReferenceCounterAccessor
{
 public:

  static ARCCORE_EXPORT void addReference(T* t);
  static ARCCORE_EXPORT void removeReference(T* t);
};
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arccore::ExternalReferenceCounterAccessor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Structure used to tag interfaces/classes that use
 * an internal reference counter.
 *
 * This tag is used via a typedef as follows:
 *
 * \code
 * class MyClass
 * {
 *   public:
 *    using ReferenceCounterTagType = ReferenceCounterTag;
 *   public:
 *    void addReference();
 *    void removeReference();
 * };
 * \endcode
 */
struct ReferenceCounterTag
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

constexpr int REF_TAG_SHARED_PTR = 0;
constexpr int REF_TAG_REFERENCE_COUNTER = 1;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Function to determine what type of reference counter
 * a class uses.
 *
 * By default, std::shared_ptr is used.
 * To use an internal reference counter, this
 * method must be overridden using the macro ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS().
 */
inline constexpr int arcaneImplGetRefTagId(void*)
{
  return REF_TAG_SHARED_PTR;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Characteristics for managing reference counters.
 *
 * By default, the std::shared_ptr class is used as the implementation.
 */
template <typename InstanceType>
struct RefTraits
{
  static constexpr int TagId = arcaneImplGetRefTagId(static_cast<InstanceType*>(nullptr));
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename InstanceType, int TagType>
struct RefTraitsTagId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accessor for reference counter management methods.
 *
 * The class T must define two methods addReference() and removeReference()
 * to manage reference counters. removeReference() must destroy
 * the instance if the counter reaches zero.
 */
template <class T>
class ReferenceCounterAccessor
{
 public:

  static void addReference(T* t)
  {
    if constexpr (requires { t->_internalAddReference(); })
      t->_internalAddReference();
    else
      t->addReference();
  }
  static void removeReference(T* t)
  {
    if constexpr (requires { t->_internalRemoveReference(); }) {
      bool need_destroy = t->_internalRemoveReference();
      if (need_destroy)
        delete t;
    }
    else
      t->removeReference();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to declare the virtual methods managing reference
 * counters.
 *
 * This macro is used in the same way as declarations
 * of interface methods. It allows defining pure virtual methods
 * to access reference counter information.
 *
 * The class implementing the interface must use the macro
 * ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS() to define the
 * virtual methods used.
 *
 * \code
 * class IMyInterface
 * {
 *   ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
 *  public:
 *   virtual ~IMyInterface() = default;
 *  public:
 *   virtual void myMethod1() = 0;
 * };
 * \endcode
 */
#define ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS() \
 private: \
\
  template <typename T> friend class ::Arccore::ExternalReferenceCounterAccessor; \
  template <typename T> friend class Arcane::ReferenceCounterAccessor; \
\
 public: \
\
  using ReferenceCounterTagType = ::Arcane::ReferenceCounterTag; \
  virtual ::Arcane::ReferenceCounterImpl* _internalReferenceCounter() = 0; \
  virtual void _internalAddReference() = 0; \
  [[nodiscard]] virtual bool _internalRemoveReference() = 0
// NOTE: The 'friend' classes are necessary for access to the destructor.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to declare that a class uses a
 * reference counter.
 *
 * The macro must be used outside of any namespace. For example:
 *
 * \code
 * namespace MyNamespace
 * {
 *   class MyClass;
 * }
 *
 * ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(MyNamespace::MyClass);
 * \endcode
 *
 * You will then need to use the macro
 * ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS() in the source file to
 * define the necessary methods and types
 */
#define ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(class_name) \
  namespace Arcane \
  { \
    template <> \
    struct RefTraits<class_name> \
    { \
      static constexpr int TagId = ::Arcane::REF_TAG_REFERENCE_COUNTER; \
    }; \
    constexpr inline int arcaneImplGetRefTagId(class_name*) \
    { \
      return ::Arcane::REF_TAG_REFERENCE_COUNTER; \
    } \
    template <> \
    class ReferenceCounterAccessor<class_name> \
    : public ExternalReferenceCounterAccessor<class_name> \
    {}; \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::ReferenceCounterTag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
