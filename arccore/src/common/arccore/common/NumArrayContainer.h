// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayContainer.h                                         (C) 2000-2026 */
/*                                                                           */
/* Container for the 'NumArray' class.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_NUMARRAYCONTAINER_H
#define ARCCORE_COMMON_NUMARRAYCONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Common implementation for NumArray.
 */
class ARCCORE_COMMON_EXPORT NumArrayBaseCommon
{
 protected:

  static MemoryAllocationOptions _getDefaultAllocator();
  static MemoryAllocationOptions _getDefaultAllocator(eMemoryResource r);
  static void _checkHost(eMemoryResource r);
  static void _memoryAwareCopy(Span<const std::byte> from, eMemoryResource from_mem,
                               Span<std::byte> to, eMemoryResource to_mem, const RunQueue* queue);
  static void _memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                               Int32 datatype_size, SmallSpan<const Int32> indexes, const RunQueue* queue);
  static void _memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                               Int32 datatype_size, const RunQueue* queue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Container for the NumArray class.
 *
 * Wrapper of Arccore::Array for the NumArray class.
 */
template <typename DataType>
class NumArrayContainer
: private Array<DataType>
, private NumArrayBaseCommon
{
 private:

  using BaseClass = Array<DataType>;
  using ThatClass = NumArrayContainer<DataType>;
  static constexpr Int32 _typeSize() { return static_cast<Int32>(sizeof(DataType)); }

 public:

  using BaseClass::capacity;
  using BaseClass::debugName;
  using BaseClass::fill;
  using BaseClass::setDebugName;

 private:

  explicit NumArrayContainer(const MemoryAllocationOptions& a)
  : BaseClass()
  {
    this->_initFromAllocator(a, 0);
  }

 public:

  explicit NumArrayContainer()
  : NumArrayContainer(_getDefaultAllocator())
  {
  }

  explicit NumArrayContainer(eMemoryResource r)
  : NumArrayContainer(_getDefaultAllocator(r))
  {
    m_memory_ressource = r;
  }

  NumArrayContainer(const ThatClass& rhs)
  : NumArrayContainer(rhs.allocationOptions())
  {
    m_memory_ressource = rhs.m_memory_ressource;
    _resizeAndCopy(rhs);
  }

  NumArrayContainer(ThatClass&& rhs)
  : BaseClass(std::move(rhs))
  , m_memory_ressource(rhs.m_memory_ressource)
  {
  }

  // This operator is deleted because it requires handling potential
  // allocator changes and copying
  ThatClass& operator=(const ThatClass& rhs) = delete;

  ThatClass& operator=(ThatClass&& rhs)
  {
    this->_move(rhs);
    m_memory_ressource = rhs.m_memory_ressource;
    return (*this);
  }

 public:

  void resize(Int64 new_size) { BaseClass::_resizeNoInit(new_size); }
  Span<DataType> to1DSpan() { return BaseClass::span(); }
  Span<const DataType> to1DSpan() const { return BaseClass::constSpan(); }
  Span<std::byte> bytes() { return asWritableBytes(BaseClass::span()); }
  Span<const std::byte> bytes() const { return asBytes(BaseClass::constSpan()); }
  void swap(NumArrayContainer<DataType>& rhs)
  {
    BaseClass::_swap(rhs);
    std::swap(m_memory_ressource, rhs.m_memory_ressource);
  }
  void copy(Span<const DataType> rhs) { BaseClass::_copy(rhs.data()); }
  IMemoryAllocator* allocator() const { return BaseClass::allocator(); }
  //TODO: deprecate by mid 2026
  eMemoryResource memoryRessource() const { return m_memory_ressource; }
  eMemoryResource memoryResource() const { return m_memory_ressource; }
  void copyInitializerList(std::initializer_list<DataType> alist)
  {
    Span<DataType> s = to1DSpan();
    Int64 s1 = s.size();
    Int32 index = 0;
    for (auto x : alist) {
      s[index] = x;
      ++index;
      // Ensures no overflow
      if (index >= s1)
        break;
    }
  }

  /*!
   * \brief Copies the values of \a v into the instance.
   *
   * \a input_ressource indicates the origin of the memory area
   * (or eMemoryRessource::Unknown if unknown)
   */
  void copyOnly(const Span<const DataType>& v, eMemoryResource input_ressource, const RunQueue* queue = nullptr)
  {
    _memoryAwareCopy(v, input_ressource, queue);
  }
  /*!
   * \brief Fills the indices provided by \a indexes with the value \a v.
   */
  void fill(const DataType& v, SmallSpan<const Int32> indexes, const RunQueue* queue)
  {
    Span<DataType> destination = to1DSpan();
    NumArrayBaseCommon::_memoryAwareFill(asWritableBytes(destination), destination.size(), &v, _typeSize(), indexes, queue);
  }
  /*!
   * \brief Fills the elements of the instance with the value \a v.
   */
  void fill(const DataType& v, const RunQueue* queue)
  {
    Span<DataType> destination = to1DSpan();
    NumArrayBaseCommon::_memoryAwareFill(asWritableBytes(destination), destination.size(), &v, _typeSize(), queue);
  }

 private:

  void _memoryAwareCopy(const Span<const DataType>& v, eMemoryResource input_ressource, const RunQueue* queue)
  {
    NumArrayBaseCommon::_memoryAwareCopy(asBytes(v), input_ressource,
                                         asWritableBytes(to1DSpan()), m_memory_ressource, queue);
  }
  void _resizeAndCopy(const ThatClass& v)
  {
    this->_resizeNoInit(v.to1DSpan().size());
    _memoryAwareCopy(v, v.memoryRessource(), nullptr);
  }

 private:

  eMemoryResource m_memory_ressource = eMemoryResource::UnifiedMemory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
