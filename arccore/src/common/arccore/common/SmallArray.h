// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SmallArray.h                                                (C) 2000-2025 */
/*                                                                           */
/* 1D data array with pre-allocated buffer.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_SMALLARRAY_H
#define ARCCORE_COMMON_SMALLARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"
#include "arccore/common/IMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Allocator with pre-allocated buffer.
 *
 * The pre-allocated buffer \a m_preallocated_buffer is used when the
 * requested size for the allocation is less than or equal to
 * \a m_preallocated_size.
 *
 * The buffer used must remain valid throughout the lifetime of the allocator.
 */
class ARCCORE_COMMON_EXPORT StackMemoryAllocator final
: public IMemoryAllocator
{
 public:

  StackMemoryAllocator(void* buf, size_t size)
  : m_preallocated_buffer(buf)
  , m_preallocated_size(size)
  {}

 public:

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate([[maybe_unused]] MemoryAllocationArgs args, Int64 new_size) final
  {
    if (new_size <= m_preallocated_size) {
      return { m_preallocated_buffer, new_size };
    }
    return { _allocateMemory(new_size), new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) final;
  void deallocate([[maybe_unused]] MemoryAllocationArgs args, AllocatedMemoryInfo ptr_info) final
  {
    void* ptr = ptr_info.baseAddress();
    if (ptr != m_preallocated_buffer)
      _freeMemory(ptr);
  }
  Int64 adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64) const final { return wanted_capacity; }
  size_t guaranteedAlignment(MemoryAllocationArgs) const final { return 0; }

 private:

  void* m_preallocated_buffer = nullptr;
  Int64 m_preallocated_size = 0;

 private:

  void* _allocateMemory(Int64 size);
  void _freeMemory(void* pointer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief 1D data array with pre-allocated stack buffer.
 *
 * This class is used like UniqueArray but contains a fixed-size buffer to
 * hold \a NbElement elements, which is used if the array contains at most
 * \a NbElement elements. This avoids dynamic allocations when the number of elements is small.
 *
 * If the array must contain more than \a NbElement elements, then standard
 * dynamic allocation is used.
 */
template <typename T, Int32 NbElement = 32>
class SmallArray final
: public Array<T>
{
  using BaseClassType = AbstractArray<T>;
  static constexpr Int32 SizeOfType = static_cast<Int32>(sizeof(T));
  static constexpr Int32 nb_element_in_buf = NbElement;

 public:

  using typename BaseClassType::ConstReferenceType;
  static constexpr Int32 MemorySize = NbElement * SizeOfType;

 public:

  //! Creates an empty array
  SmallArray()
  : m_stack_allocator(m_stack_buffer, MemorySize)
  {
    this->_initFromAllocator(MemoryAllocationOptions(&m_stack_allocator), nb_element_in_buf, m_stack_buffer);
  }

  //! Creates an array of \a size elements containing the value \a value.
  SmallArray(Int64 req_size, ConstReferenceType value)
  : SmallArray()
  {
    this->_resize(req_size, value);
  }

  //! Creates an array of \a asize elements containing the default value of type T()
  explicit SmallArray(Int64 asize)
  : SmallArray()
  {
    this->_resize(asize);
  }

  //! Creates an array of \a asize elements containing the default value of type T()
  explicit SmallArray(Int32 asize)
  : SmallArray((Int64)asize)
  {
  }

  //! Creates an array of \a asize elements containing the default value of type T()
  explicit SmallArray(size_t asize)
  : SmallArray((Int64)asize)
  {
  }

  //! Creates an array by copying the values from the view \a aview.
  SmallArray(const ConstArrayView<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  //! Creates an array by copying the values from the view \a aview.
  SmallArray(const Span<const T>& aview)
  : SmallArray()
  {
    this->_initFromSpan(aview);
  }

  //! Creates an array by copying the values from the view \a aview.
  SmallArray(const ArrayView<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  //! Creates an array by copying the values from the view \a aview.
  SmallArray(const Span<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  SmallArray(std::initializer_list<T> alist)
  : SmallArray()
  {
    this->_initFromInitializerList(alist);
  }

  //! Creates an array by copying the values \a rhs.
  SmallArray(const Array<T>& rhs)
  : SmallArray(rhs.constSpan())
  {
  }

  //! Copies the values of \a rhs into this instance.
  void operator=(const Array<T>& rhs)
  {
    this->copy(rhs.constSpan());
  }

  //! Copies the values of the view \a rhs into this instance.
  void operator=(const ArrayView<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copies the values of the view \a rhs into this instance.
  void operator=(const Span<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copies the values of the view \a rhs into this instance.
  void operator=(const ConstArrayView<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copies the values of the view \a rhs into this instance.
  void operator=(const Span<const T>& rhs)
  {
    this->copy(rhs);
  }

  //! Destroys the instance.
  ~SmallArray() override
  {
    // It must be explicitly destroyed because our allocator
    // will be destroyed before the base class and will no longer be valid
    // during deallocation.
    this->_destroy();
    this->_internalDeallocate();
    this->_reset();
  }

 public:

  template <Int32 N> SmallArray(SmallArray<T, N>&& rhs) = delete;
  template <Int32 N> SmallArray<T, NbElement> operator=(SmallArray<T, N>&& rhs) = delete;

 private:

  char m_stack_buffer[MemorySize];
  Impl::StackMemoryAllocator m_stack_allocator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
