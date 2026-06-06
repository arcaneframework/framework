// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupObserver.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface and basic implementation of group observers.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGROUPOBSERVER_H
#define ARCANE_CORE_ITEMGROUPOBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemGroupObserver
{
 public:

  template <typename T>
  struct FuncTraits
  {
    //! Type of the pointer to the method with Info
    typedef void (T::*FuncPtrWithInfo)(const Int32ConstArrayView* info);

    //! Type of the pointer to the method without Info
    typedef void (T::*FuncPtr)();
  };

 public:

  //! Destructor
  virtual ~IItemGroupObserver() = default;

  /*!
   * \brief Execute the action associated with the extension.
   *
   * \param info list of added localIds
   * Assumes there is no change in order or renumbering.
   *
   * This method cannot be parallel.
   */
  virtual void executeExtend(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Execute the action associated with the extension.
   *
   * \param info list of positions removed in the old group
   * Assumes there is no change in order or renumbering
   * This approach compared to the list of localIds is motivated by
   * the constraint in PartialVariable which is unaware of the localIds
   * it hosts.
   * \param info2 list of localIds of deleted elements. Potentially redundant
   * with \a info, but inevitable for certain structures changing the order relative to
   * the reference group (e.g.: ItemGroupDynamicMeshObserver) (DEPRECATED)
   *
   * This method cannot be parallel.
   */
  virtual void executeReduce(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Executes the action associated with compaction.
   *
   * \param info list of permutations in the old->new direction
   * Assumes there is no change in size.
   */
  virtual void executeCompact(const Int32ConstArrayView* info) = 0;

  /*!
   * \brief Execute the action associated with invalidation.
   *
   * No transition information available.
   */
  virtual void executeInvalidate() = 0;

  /*!
   * \brief Indicates whether the observer will need transition information
   *
   * This information must not change after the first call to this function
   */
  virtual bool needInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ItemGroupObserverWithInfoT
: public IItemGroupObserver
{
 public:

  ItemGroupObserverWithInfoT(T* object,
                             typename FuncTraits<T>::FuncPtrWithInfo extend_funcptr,
                             typename FuncTraits<T>::FuncPtrWithInfo reduce_funcptr,
                             typename FuncTraits<T>::FuncPtrWithInfo compact_funcptr,
                             typename FuncTraits<T>::FuncPtr invalidate_funcptr)
  : m_object(object)
  , m_extend_function(extend_funcptr)
  , m_reduce_function(reduce_funcptr)
  , m_compact_function(compact_funcptr)
  , m_invalidate_function(invalidate_funcptr)
  {}

 public:

  void executeExtend(const Int32ConstArrayView* info) override
  {
    (m_object->*m_extend_function)(info);
  }

  void executeReduce(const Int32ConstArrayView* info) override
  {
    (m_object->*m_reduce_function)(info);
  }

  void executeCompact(const Int32ConstArrayView* info) override
  {
    (m_object->*m_compact_function)(info);
  }

  void executeInvalidate() override
  {
    (m_object->*m_invalidate_function)();
  }

  bool needInfo() const override { return true; }

 private:

  T* m_object = nullptr; //!< Associated object.
  typename FuncTraits<T>::FuncPtrWithInfo m_extend_function = nullptr; //!< Pointer to the associated method.
  typename FuncTraits<T>::FuncPtrWithInfo m_reduce_function = nullptr; //!< Pointer to the associated method.
  typename FuncTraits<T>::FuncPtrWithInfo m_compact_function = nullptr; //!< Pointer to the associated method.
  typename FuncTraits<T>::FuncPtr m_invalidate_function = nullptr; //!< Pointer to the associated method.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ItemGroupObserverWithoutInfoT
: public IItemGroupObserver
{
 public:

  //! Constructor from a single argument-less function
  ItemGroupObserverWithoutInfoT(T* object, typename FuncTraits<T>::FuncPtr funcptr)
  : m_object(object)
  , m_function(funcptr)
  {}

 public:

  void executeExtend(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeReduce(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeCompact(const Int32ConstArrayView*) override
  {
    (m_object->*m_function)();
  }

  void executeInvalidate() override
  {
    (m_object->*m_function)();
  }

  bool needInfo() const override
  {
    return false;
  }

 private:

  T* m_object = nullptr; //!< Associated object.
  typename FuncTraits<T>::FuncPtr m_function = nullptr; //!< Pointer to the associated method.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Utility for simplified creation of ItemGroupObserverT
template <typename T>
inline IItemGroupObserver* newItemGroupObserverT(T* object,
                                                 typename IItemGroupObserver::FuncTraits<T>::FuncPtr funcptr)
{
  return new ItemGroupObserverWithoutInfoT<T>(object, funcptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Utility for simplified creation of ItemGroupObserverT
template <typename T> inline IItemGroupObserver*
newItemGroupObserverT(T* object,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo extend_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo reduce_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtrWithInfo compact_funcptr,
                      typename IItemGroupObserver::FuncTraits<T>::FuncPtr invalidate_funcptr)
{
  return new ItemGroupObserverWithInfoT<T>(object, extend_funcptr, reduce_funcptr, compact_funcptr, invalidate_funcptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
