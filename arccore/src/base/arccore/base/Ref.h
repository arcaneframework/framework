// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Ref.h                                                       (C) 2000-2026 */
/*                                                                           */
/* Instance reference management.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REF_H
#define ARCCORE_BASE_REF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RefDeclarations.h"
#include "arccore/base/RefBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*!
 * \brief Wrapper around a class managing its own reference counter.
 *
 * The class \a InstanceType must manage its own reference counter and
 * its own destruction.
 */
template <typename InstanceType>
class ReferenceCounterWrapper
{
  //! Checks that the class 'InstanceType' has a typedef for
  //! ReferenceCounterTag (\sa Ref)
  inline static void _checkHasReferenceCounterTag()
  {
    static_assert(std::is_same_v<typename InstanceType::ReferenceCounterTagType, ReferenceCounterTag>, "Bad tag");
  }

 public:

  //! Constructor with an empty deleter. In this case, it does not need to be kept
  ReferenceCounterWrapper(InstanceType* ptr, const RefBase::BasicDeleterBase&)
  : m_instance(ptr)
  {
    _checkHasReferenceCounterTag();
  }
  template <typename U> // U must derive from \a RefBase::DeleterBase
  ReferenceCounterWrapper(InstanceType* ptr, const U& deleter_base)
  : m_instance(ptr)
  {
    // This constructor is called when there is an associated ExternalRef or if we specify
    // that the object must be destroyed manually.
    _checkHasReferenceCounterTag();
    m_instance->_internalReferenceCounter()->_setExternalDeleter(new RefBase::DeleterBase(deleter_base));
  }
  explicit ReferenceCounterWrapper(InstanceType* ptr)
  : m_instance(ptr)
  {
    _checkHasReferenceCounterTag();
  }
  //! Allows conversion if 'T*' and 'InstanceType*' are convertible
  template <typename T,
            typename X = typename std::is_convertible<T*, InstanceType*>::type>
  explicit ReferenceCounterWrapper(const ReferenceCounterWrapper<T>& r)
  : m_instance(r.get())
  {
    _checkHasReferenceCounterTag();
  }
  ReferenceCounterWrapper() = default;

 public:

  //! Returns the instance
  InstanceType* get() const { return m_instance.get(); }

  //! Resets the currently associated reference.
  void reset() { m_instance = nullptr; }

  RefBase::DeleterBase* getDeleter()
  {
    return m_instance.get()->_internalReferenceCounter()->_externalDeleter();
  }

 private:

  Arccore::ReferenceCounter<InstanceType> m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization to indicate that 'shared_ptr' implementation is used
template <typename InstanceType>
struct RefTraitsTagId<InstanceType, REF_TAG_SHARED_PTR>
{
  using ImplType = std::shared_ptr<InstanceType>;
  static constexpr int RefType = REF_TAG_SHARED_PTR;
};

//! Specialization to indicate that 'ReferenceCounter' implementation is used
template <typename InstanceType>
struct RefTraitsTagId<InstanceType, REF_TAG_REFERENCE_COUNTER>
{
  using ImplType = impl::ReferenceCounterWrapper<InstanceType>;
  static constexpr int RefType = REF_TAG_REFERENCE_COUNTER;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of the reference to an instance.
 *
 * \sa Ref
 */
template <typename InstanceType, typename RefClassType, int ImplTagId>
class RefImpl
: public RefBase
{
  // NOTE: RefClassType is used only to access the destructor
  // of \a InstanceType which may be private and 'friend' of 'Ref'.

 public:

  using ThatClass = RefImpl<InstanceType, RefClassType, ImplTagId>;
  using ImplType = RefTraitsTagId<InstanceType, ImplTagId>::ImplType;
  template <typename T> friend class Ref;

 private:

  template <typename... _Args>
  using _IsRefConstructible = typename std::enable_if<std::is_constructible<ImplType, _Args...>::value>::type;

  using RefBase::DeleterBase;
  class Deleter : public DeleterBase
  {
   public:

    Deleter() = default;
    Deleter(Internal::ExternalRef h)
    : DeleterBase(h)
    {}
    Deleter(Internal::ExternalRef h, bool no_destroy)
    : DeleterBase(std::move(h), no_destroy)
    {}
    void operator()(InstanceType* tt)
    {
      if (m_no_destroy)
        return;
      bool is_destroyed = this->_destroyHandle(tt, m_handle);
      if (!is_destroyed)
        RefClassType::_destroyInstance(tt);
    }
  };

  class BasicDeleter
  : public BasicDeleterBase
  {
   public:

    bool hasExternal() const { return false; }
    void operator()(InstanceType* tt)
    {
      RefClassType::_destroyInstance(tt);
    }
  };

 public:

  static constexpr int RefType = RefTraitsTagId<InstanceType, ImplTagId>::RefType;

 private:

  explicit RefImpl(InstanceType* t)
  : m_instance(t, _createBasicDeleter((ImplType*)nullptr)) //BasicDeleter{})
  {}
  RefImpl(InstanceType* t, Internal::ExternalRef handle)
  : m_instance(t, Deleter(handle))
  {}
  RefImpl(InstanceType* t, bool no_destroy)
  : m_instance(t, Deleter(nullptr, no_destroy))
  {}

 private:

  RefImpl(ImplType&& t)
  : m_instance(t)
  {}

 public:

  /*!
   * \brief Constructs a reference from another reference of a compatible type.
   *
   * Conversion is allowed if an instance of 'ImplType'
   * can be constructed from an instance of Ref<T>::ImplType.
   */
  template <typename T, typename = _IsRefConstructible<typename RefImpl<T, Ref<T>, ImplTagId>::ImplType>>
  explicit RefImpl(const Ref<T>& rhs) noexcept
  : m_instance(rhs._internalInstance())
  {}
  RefImpl() = default;
  RefImpl(const ThatClass& rhs) = default;
  ThatClass& operator=(const ThatClass& rhs) = default;
  ~RefImpl() = default;

 public:
 public:

  friend inline bool operator==(const ThatClass& a, const ThatClass& b)
  {
    return a.get() == b.get();
  }

  friend inline bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return a.get() != b.get();
  }

  friend inline bool operator<(const ThatClass& a, const ThatClass& b)
  {
    return a.get() < b.get();
  }

  friend inline bool operator!(const ThatClass& a)
  {
    return a.isNull();
  }

  operator bool() const { return (!isNull()); }

 public:

  //! Associated instance or `nullptr` if none
  InstanceType* get() const { return m_instance.get(); }
  //! Indicates if the counter references a non-null instance.
  bool isNull() const { return m_instance.get() == nullptr; }
  InstanceType* operator->() const { return m_instance.get(); }
  //! Positions the instance to the null pointer.
  void reset() { m_instance.reset(); }
  /*!
   * \internal
   * \brief Releases the reference counter pointer without destroying it.
   * This is only allowed if the implementation uses 'std::shared_ptr'.
   */
  template <typename T = ThatClass, typename std::enable_if_t<T::RefType == REF_TAG_SHARED_PTR, bool> = true>
  InstanceType* _release()
  {
    // Releases the instance. To do this, we indicate to the destructor
    // not to destroy the instance of 'm_instance' and we
    // return the latter.
    auto* r = _getDeleter(m_instance);
    if (r)
      r->setNoDestroy(true);
    InstanceType* t = m_instance.get();
    m_instance.reset();
    return t;
  }
  const ImplType& _internalInstance() const { return m_instance; }

 private:

  ImplType m_instance;

 private:

  static Deleter _createBasicDeleter(std::shared_ptr<InstanceType>*)
  {
    return {};
  }
  static BasicDeleterBase _createBasicDeleter(impl::ReferenceCounterWrapper<InstanceType>*)
  {
    return {};
  }
  static Deleter* _getDeleter(std::shared_ptr<InstanceType>& v)
  {
    return std::get_deleter<Deleter>(v);
  }
  static DeleterBase* _getDeleter(impl::ReferenceCounterWrapper<InstanceType>& v)
  {
    return v.getDeleter();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reference to an instance.
 *
 * This class uses a reference counter to manage the lifetime
 * of a C++ instance. It works similarly to std::shared_ptr.
 *
 * When the last instance of this class is destroyed, the referenced
 * instance is destroyed. The way the associated instance is destroyed
 * is specified when creating the first reference via a call
 * to one of the methods create() or createWithHandle().
 *
 * There are two possible implementations for counting references.
 * By default, 'std::shared_ptr' is used. It is also possible
 * to use an internal reference counter in the class, which
 * allows compatibility with the ReferenceCounter class and also allows
 * retrieving a reference from the instance itself. This
 * second implementation is accessible by specializing the type
 * RefTraits so that it defines a ReferenceCounterTagType
 * equal to ReferenceCounterTag.
 */
template <typename InstanceType>
class Ref
: public RefImpl<InstanceType, Ref<InstanceType>, RefTraits<InstanceType>::TagId>
{
  using Base = RefImpl<InstanceType, Ref<InstanceType>, RefTraits<InstanceType>::TagId>;
  using ImplType = Base::ImplType;
  using ThatClass = Ref<InstanceType>;
  friend Base;

  template <typename... _Args>
  using _IsRefConstructible = typename std::enable_if<std::is_constructible<ImplType, _Args...>::value>::type;

 private:

  explicit Ref(InstanceType* t)
  : Base(t)
  {}
  Ref(InstanceType* t, Internal::ExternalRef handle)
  : Base(t, handle)
  {}
  Ref(InstanceType* t, bool no_destroy)
  : Base(t, no_destroy)
  {}

 public:

  /*!
   * \brief Constructs a reference from another reference of a compatible type.
   *
   * Conversion is allowed if an instance of 'ImplType'
   * can be constructed
   * from an instance of Ref<T>::ImplType.
   */
  template <typename T> //, typename = _IsRefConstructible<typename Ref<T>::ImplType>>
  Ref(const Ref<T>& rhs) noexcept
  : Base(rhs)
  {}
  Ref() = default;
  Ref(const ThatClass& rhs) = default;
  ThatClass& operator=(const ThatClass& rhs) = default;
  ~Ref() = default;

 public:

  /*!
   * \internal
   * \brief Creates a reference from the instance \a t.
   *
   * This method is internal to %Arccore.
   *
   * The instance \a t must have been created by the 'operator new'
   * operator and will be destroyed by the 'operator delete' operator
   */
  static ThatClass create(InstanceType* t)
  {
    return ThatClass(t);
  }

  template <typename PointerType, typename... Args>
  static inline Ref<InstanceType>
  createRef(Args&&... args)
  {
    PointerType* pt = new PointerType(std::forward<Args>(args)...);
    return Ref<InstanceType>(pt);
  }

  /*!
   * \internal
   * \brief Creates a reference from an instance having an
   * external reference.
   */
  static ThatClass createWithHandle(InstanceType* t, Internal::ExternalRef handle)
  {
    return ThatClass(t, handle);
  }

  static ThatClass _createNoDestroy(InstanceType* t)
  {
    return ThatClass(t, true);
  }

 private:

  static void _destroyInstance(InstanceType* t)
  {
    delete t;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a reference on a pointer.
 *
 * The pointer \a t must have been allocated by the 'operator new'
 * operator and will be destroyed by the 'operator delete' operator when there is no longer
 * a reference to it.
 */
template <typename InstanceType> inline auto
makeRef(InstanceType* t) -> Ref<InstanceType>
{
  return Ref<InstanceType>::create(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves a reference on the pointer \a t.
 *
 * This method is only available if the class InstanceType uses a
 * reference counter (ImplTagId==REF_TAG_REFERENCE_COUNTER).
 *
 * \code
 * class A {};
 * class B : public A {};
 * Ref<B> rb = ...;
 * B* b = rb.get();
 * Ref<A> ra = makeRefFromInstance<A>(b);
 * \endcode
 */
template <typename InstanceType,
          typename InstanceType2,
          typename std::enable_if_t<Ref<InstanceType>::RefType, int> = REF_TAG_REFERENCE_COUNTER>
inline Ref<InstanceType>
makeRefFromInstance(InstanceType2* t)
{
  return Ref<InstanceType>::create(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates an instance of type \a TrueType with arguments \a Args
 * and returns a reference to it.
 */
template <typename TrueType, class... Args> inline Ref<TrueType>
createRef(Args&&... args)
{
  return makeRef<TrueType>(new TrueType(std::forward<Args>(args)...));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::createRef;
using Arcane::makeRef;
using Arcane::makeRefFromInstance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
