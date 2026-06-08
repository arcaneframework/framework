// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Multi-dimensional arrays for numerical types.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_NUMARRAY_H
#define ARCCORE_COMMON_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/MDSpan.h"
#include "arccore/base/MDDim.h"
#include "arccore/base/String.h"
#include "arccore/common/NumArrayContainer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
concept NumArrayDataTypeConcept = std::is_trivially_copyable_v<T>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Multi-dimensional arrays for numerical types accessible
 * on accelerators.
 *
 * The current implementation supports arrays up to 4 dimensions. Element access
 * is done via the 'operator()'.
 *
 * \warning Resizing via resize() does not preserve existing values
 * except for rank 1 arrays.
 *
 * \warning This class by default uses a specific allocator that allows these
 * values to be accessible both on the host (CPU) and the accelerator.
 * However, this requires that the accelerator's associated runtime has been
 * initialized (\ref arcanedoc_parallel_accelerator). Therefore, global variables
 * of this class or a derived class must not be used.
 *
 * For more information, refer to the page \ref arcanedoc_core_types_numarray.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class NumArray
: private Impl::NumArrayBaseCommon
{
 public:

#if !defined(ARCANE_NO_CONCEPT_FOR_NUMARRAY)
  static_assert(NumArrayDataTypeConcept<DataType>, "concept 'NumArrayDataTypeConcept' is not fullfilled");
#endif

 public:

  using ExtentsType = Extents;
  using ThatClass = NumArray<DataType, Extents, LayoutPolicy>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;
  using ConstMDSpanType = MDSpan<const DataType, ExtentsType, LayoutPolicy>;
  using MDSpanType = MDSpan<DataType, ExtentsType, LayoutPolicy>;
  using ArrayWrapper = Impl::NumArrayContainer<DataType>;
  using ArrayBoundsIndexType = typename MDSpanType::ArrayBoundsIndexType;
  using value_type = DataType;
  using LayoutPolicyType = LayoutPolicy;

  using ConstSpanType ARCCORE_DEPRECATED_REASON("Use 'ConstMDSpanType' instead") = ConstMDSpanType;
  using SpanType ARCCORE_DEPRECATED_REASON("Use 'MDSpanType' instead") = MDSpanType;

 public:

  //! Number of dimensions of the array
  static constexpr int rank() { return Extents::rank(); }

 public:

  //! Constructs an empty array
  NumArray()
  {
    _resizeInit();
  }

  //! Constructs an array by specifying the dimension list directly
  explicit NumArray(DynamicDimsType extents)
  {
    resize(extents);
  }

  //! Constructs an array by specifying the dimension list directly
  NumArray(const DynamicDimsType& extents, eMemoryResource r)
  : m_data(r)
  {
    resize(extents);
  }
  //! Creates an empty array using the memory resource \a r
  explicit NumArray(eMemoryResource r)
  : m_data(r)
  {
    _resizeInit();
  }

  //! Constructs an array with 4 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size,
           Int32 dim3_size, Int32 dim4_size) requires(Extents::nb_dynamic == 4)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size))
  {
  }

  //! Constructs an array with 4 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size,
           Int32 dim3_size, Int32 dim4_size, eMemoryResource r) requires(Extents::nb_dynamic == 4)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size), r)
  {
  }

  //! Constructs an array with 3 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size) requires(Extents::nb_dynamic == 3)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size))
  {
  }
  //! Constructs an array with 3 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size, eMemoryResource r) requires(Extents::nb_dynamic == 3)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size), r)
  {
  }

  //! Constructs an array with 2 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size) requires(Extents::nb_dynamic == 2)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size))
  {
  }
  //! Constructs an array with 2 dynamic values
  NumArray(Int32 dim1_size, Int32 dim2_size, eMemoryResource r) requires(Extents::nb_dynamic == 2)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size), r)
  {
  }

  //! Constructs an array with 1 dynamic value
  explicit NumArray(Int32 dim1_size) requires(Extents::nb_dynamic == 1)
  : ThatClass(DynamicDimsType(dim1_size))
  {
  }
  //! Constructs an array with 1 dynamic value
  NumArray(Int32 dim1_size, eMemoryResource r) requires(Extents::nb_dynamic == 1)
  : ThatClass(DynamicDimsType(dim1_size), r)
  {
  }

  /*!
   * \brief Constructs an array from predefined values (only dynamic 2D arrays).
   *
   * The values are stored contiguously in memory, so
   * the list \a alist must have a layout corresponding to this class.
   */
  NumArray(Int32 dim1_size, Int32 dim2_size, std::initializer_list<DataType> alist)
  requires(Extents::is_full_dynamic() && Extents::rank() == 2)
  : NumArray(dim1_size, dim2_size)
  {
    this->m_data.copyInitializerList(alist);
  }

  //! Constructs an array from predefined values (only dynamic 1D arrays)
  NumArray(Int32 dim1_size, std::initializer_list<DataType> alist)
  requires(Extents::isDynamic1D())
  : NumArray(dim1_size)
  {
    this->m_data.copyInitializerList(alist);
  }

  //! Constructs an instance from a view (only dynamic 1D arrays)
  NumArray(SmallSpan<const DataType> v)
  requires(Extents::isDynamic1D())
  : NumArray(v.size())
  {
    copy(v);
  }

  //! Constructs an instance from a view (only dynamic 1D arrays)
  NumArray(Span<const DataType> v)
  requires(Extents::isDynamic1D())
  {
    copy(v.smallView());
  }

  NumArray(const ThatClass& rhs)
  : m_span(rhs.m_span)
  , m_data(rhs.m_data)
  , m_total_nb_element(rhs.m_total_nb_element)
  {
    _updateSpanPointerFromData();
  }

  NumArray(ThatClass&& rhs)
  : m_span(rhs.m_span)
  , m_data(std::move(rhs.m_data))
  , m_total_nb_element(rhs.m_total_nb_element)
  {
  }

  ThatClass& operator=(ThatClass&&) = default;

  /*!
   * \brief Copy assignment operator.
   *
   * \warning After calling this method, the instance's memory resource
   * will be that of \a rhs. If you wish to perform a copy while keeping the
   * associated memory resource, you must use copy().
   */
  ThatClass& operator=(const ThatClass& rhs)
  {
    if (&rhs == this)
      return (*this);
    eMemoryResource r = memoryResource();
    eMemoryResource rhs_r = rhs.memoryResource();
    if (rhs_r != r)
      m_data = ArrayWrapper(rhs_r);
    this->copy(rhs);
    return (*this);
  }

  /*!
   * \brief Swaps the data with \a rhs.
   *
   * \warning The memory allocator is also swapped. It is therefore
   * preferable that both NumArray instances use the same allocator
   * and the same memoryResource().
   */
  void swap(ThatClass& rhs)
  {
    m_data.swap(rhs.m_data);
    std::swap(m_span, rhs.m_span);
    std::swap(m_total_nb_element, rhs.m_total_nb_element);
  }

 public:

  //! Total number of elements in the array
  constexpr Int64 totalNbElement() const { return m_total_nb_element; }
  //! Number of dimensions
  static constexpr Int32 nbDimension() { return Extents::rank(); }
  //! Values of the dimensions
  ArrayExtents<Extents> extents() const { return m_span.extents(); }
  ArrayExtentsWithOffset<Extents, LayoutPolicy> extentsWithOffset() const
  {
    return m_span.extentsWithOffset();
  }
  Int64 capacity() const { return m_data.capacity(); }
  //TODO: deprecate by mid 2026
  eMemoryResource memoryRessource() const { return m_data.memoryResource(); }
  eMemoryResource memoryResource() const { return m_data.memoryResource(); }
  //! View as bytes
  Span<std::byte> bytes() { return asWritableBytes(to1DSpan()); }
  //! Constant view as bytes
  Span<const std::byte> bytes() const { return asBytes(to1DSpan()); }

  //! Associated memory allocator
  IMemoryAllocator* memoryAllocator() const { return m_data.allocator(); }

  /*!
   * \brief Sets the array name for debug information.
   *
   * This name can be used, for example, in listing displays.
   */
  void setDebugName(const String& str) { m_data.setDebugName(str); }

  //! Debug name (null if no name specified)
  String debugName() { return m_data.debugName(); }

 public:

  //! Value of the first dimension
  constexpr Int32 dim1Size() const requires(Extents::rank() >= 1) { return m_span.extent0(); }
  //! Value of the second dimension
  constexpr Int32 dim2Size() const requires(Extents::rank() >= 2) { return m_span.extent1(); }
  //! Value of the third dimension
  constexpr Int32 dim3Size() const requires(Extents::rank() >= 3) { return m_span.extent2(); }
  //! Value of the fourth dimension
  constexpr Int32 dim4Size() const requires(Extents::rank() >= 4) { return m_span.extent3(); }

  //! Value of the first dimension
  constexpr Int32 extent0() const requires(Extents::rank() >= 1) { return m_span.extent0(); }
  //! Value of the second dimension
  constexpr Int32 extent1() const requires(Extents::rank() >= 2) { return m_span.extent1(); }
  //! Value of the third dimension
  constexpr Int32 extent2() const requires(Extents::rank() >= 3) { return m_span.extent2(); }
  //! Value of the fourth dimension
  constexpr Int32 extent3() const requires(Extents::rank() >= 4) { return m_span.extent3(); }

 public:

  //! Resizes the array without keeping current values
  void resize(Int32 dim1_size) requires(Extents::nb_dynamic == 1)
  {
    m_span.m_extents = DynamicDimsType(dim1_size);
    _resize();
  }

  // TODO: Deprecate (June 2025)
  //! Resizes the array without keeping current values
  void resize(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size, Int32 dim4_size) requires(Extents::nb_dynamic == 4)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size));
  }

  // TODO: Deprecate (June 2025)
  //! Resizes the array without keeping current values
  void resize(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size) requires(Extents::nb_dynamic == 3)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size, dim3_size));
  }

  // TODO: Deprecate (June 2025)
  //! Resizes the array without keeping current values
  void resize(Int32 dim1_size, Int32 dim2_size) requires(Extents::nb_dynamic == 2)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size));
  }

  /*!
   * \brief Resizes the array.
   * \warning Current values are not preserved during this operation
   * and new values are not initialized.
   */
  //@{
  //! Resizes the array without keeping current values
  void resizeDestructive(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size, Int32 dim4_size) requires(Extents::nb_dynamic == 4)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size));
  }

  //! Resizes the array without keeping current values
  void resizeDestructive(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size) requires(Extents::nb_dynamic == 3)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size, dim3_size));
  }

  //! Resizes the array without keeping current values
  void resizeDestructive(Int32 dim1_size, Int32 dim2_size) requires(Extents::nb_dynamic == 2)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size, dim2_size));
  }

  //! Resizes the array without keeping current values
  void resizeDestructive(Int32 dim1_size) requires(Extents::nb_dynamic == 1)
  {
    this->resizeDestructive(DynamicDimsType(dim1_size));
  }

  // TODO: Deprecate (June 2025)
  //! Resizes the array without keeping current values
  void resize(const DynamicDimsType& dims)
  {
    resizeDestructive(dims);
  }

  //! Resizes the array without keeping current values
  void resizeDestructive(const DynamicDimsType& dims)
  {
    m_span.m_extents = dims;
    _resize();
  }
  //@}

 public:

  /*!
   * \brief Fills the array values with \a v.
   *
   * \warning The operation is performed on the host, so the memory
   * associated with the instance must be accessible on the host.
   */
  void fill(const DataType& v)
  {
    fillHost(v);
  }

  /*!
   * \brief Fills the index array values using the queue \a queue
   * with the value \a v.
   *
   * The memory associated with the instance must be accessible
   * from the queue \a queue. \a queue can be null, in which case the
   * filling is done on the host.
   */
  void fill(const DataType& v, SmallSpan<const Int32> indexes, const RunQueue* queue)
  {
    m_data.fill(v, indexes, queue);
  }

  /*!
   * \brief Fills the index array values using the queue \a queue
   * with the value \a v.
   *
   * The memory associated with the instance must be accessible
   * from the queue \a queue.
   */
  void fill(const DataType& v, SmallSpan<const Int32> indexes, const RunQueue& queue)
  {
    m_data.fill(v, indexes, &queue);
  }

  /*!
   * \brief Fills the instance elements with the value \a v using
   * the queue \a queue.
   *
   * \a queue can be null, in which case the filling is done on
   * the host.
   */
  void fill(const DataType& v, const RunQueue* queue)
  {
    m_data.fill(v, queue);
  }

  /*!
   * \brief Fills the instance elements with the value \a v using
   * the queue \a queue.
   *
   * \a queue can be null, in which case the filling is done on
   * the host.
   */
  void fill(const DataType& v, const RunQueue& queue)
  {
    m_data.fill(v, &queue);
  }

  /*!
   * \brief Fills the array values with \a v.
   *
   * The operation is performed on the host, so the memory associated
   * with the instance must be accessible on the host.
   */
  void fillHost(const DataType& v)
  {
    _checkHost(memoryRessource());
    m_data.fill(v);
  }

 public:

  /*!
   * \brief Copies the values from \a rhs into the instance.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   */
  void copy(SmallSpan<const DataType> rhs) requires(Extents::isDynamic1D())
  {
    copy(rhs, nullptr);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   */
  void copy(ConstMDSpanType rhs) { copy(rhs, nullptr); }

  /*!
   * \brief Copies the values from \a rhs into the instance.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   */
  void copy(const ThatClass& rhs) { copy(rhs, nullptr); }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null. If the queue is asynchronous, it must be
   * synchronized before the instance can be used.
   */
  void copy(SmallSpan<const DataType> rhs, const RunQueue* queue) requires(Extents::isDynamic1D())
  {
    _resizeAndCopy(ConstMDSpanType(rhs), eMemoryResource::Unknown, queue);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null. If the queue is asynchronous, it must be
   * synchronized before the instance can be used.
   */
  void copy(ConstMDSpanType rhs, const RunQueue* queue)
  {
    _resizeAndCopy(rhs, eMemoryResource::Unknown, queue);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null, in which case the copy is performed on the host.
   * If the queue is asynchronous, it must be synchronized before the instance
   * can be used.
   */
  void copy(SmallSpan<const DataType> rhs, const RunQueue& queue) requires(Extents::isDynamic1D())
  {
    _resizeAndCopy(ConstMDSpanType(rhs), eMemoryResource::Unknown, &queue);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null, in which case the copy is performed on the host.
   * If the queue is asynchronous, it must be synchronized before the
   * instance can be used.
   */
  void copy(ConstMDSpanType rhs, const RunQueue& queue)
  {
    _resizeAndCopy(rhs, eMemoryResource::Unknown, &queue);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null, in which case the copy is performed on the host.
   * If the queue is asynchronous, it must be synchronized before the
   * instance can be used.
   */
  void copy(const ThatClass& rhs, const RunQueue* queue)
  {
    _resizeAndCopy(rhs.constMDSpan(), rhs.memoryResource(), queue);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance via the
   * queue \a queue.
   *
   * This operation is valid regardless of the memory associated
   * with the instance.
   * \a queue can be null. If the queue is asynchronous, it must be
   * synchronized before the instance can be used.
   */
  void copy(const ThatClass& rhs, const RunQueue& queue)
  {
    _resizeAndCopy(rhs.constMDSpan(), rhs.memoryRessource(), &queue);
  }

 public:

  //! Retrieves a reference for element \a i
  DataType& operator[](Int32 i) requires(Extents::rank() == 1) { return m_span(i); }
  //! Value for element \a i
  DataType operator[](Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }

 public:

  //! Value for element \a i,j,k,l
  DataType operator()(Int32 i, Int32 j, Int32 k, Int32 l) const requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }
  //! Positions the value for element \a i,j,k,l
  DataType& operator()(Int32 i, Int32 j, Int32 k, Int32 l) requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }

  //! Value for element \a i,j,k
  DataType operator()(Int32 i, Int32 j, Int32 k) const requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }
  //! Positions the value for element \a i,j,k
  DataType& operator()(Int32 i, Int32 j, Int32 k) requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }

  //! Value for element \a i,j
  DataType operator()(Int32 i, Int32 j) const requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Positions the value for element \a i,j
  DataType& operator()(Int32 i, Int32 j) requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Value for element \a i
  DataType operator()(Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }
  //! Positions the value for element \a i
  DataType& operator()(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }

 public:

  //! Constant reference for element \a idx
  const DataType& operator()(ArrayBoundsIndexType idx) const
  {
    return m_span(idx);
  }
  //! Modifiable reference for element \a idx
  DataType& operator()(ArrayBoundsIndexType idx)
  {
    return m_span(idx);
  }

 public:

  // TODO: deprecate
  //! Positions the value for element \a i,j,k,l
  ARCCORE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j, Int32 k, Int32 l) requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }
  //! Positions the value for element \a i,j,k
  ARCCORE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j, Int32 k) requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }
  //! Positions the value for element \a i,j
  ARCCORE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j) requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Positions the value for element \a i
  ARCCORE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }

  //! Positions the value for element \a idx
  ARCCORE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(ArrayBoundsIndexType idx)
  {
    return m_span(idx);
  }

 public:

  //! Multi-dimensional view on the instance
  ARCCORE_DEPRECATED_REASON("Y2024: Use mdspan() instead")
  MDSpanType span() { return m_span; }

  //! Constant multi-dimensional view on the instance
  ARCCORE_DEPRECATED_REASON("Y2024: Use mdspan() instead")
  ConstMDSpanType span() const { return m_span.constMDSpan(); }

  //! Constant multi-dimensional view on the instance
  ARCCORE_DEPRECATED_REASON("Y2024: Use constMDSpan() instead")
  ConstMDSpanType constSpan() const { return m_span.constMDSpan(); }

  //! Multi-dimensional view on the instance
  MDSpanType mdspan() { return m_span; }

  //! Constant multi-dimensional view on the instance
  ConstMDSpanType mdspan() const { return m_span.constMDSpan(); }

  //! Constant multi-dimensional view on the instance
  ConstMDSpanType constMDSpan() const { return m_span.constMDSpan(); }

  //! Constant 1D view on the instance
  Span<const DataType> to1DSpan() const { return m_span.to1DSpan(); }

  //! 1D view on the instance
  Span<DataType> to1DSpan() { return m_span.to1DSpan(); }

  //! Conversion to a multi-dimensional view on the instance
  constexpr operator MDSpanType() { return this->mdspan(); }
  //! Conversion to a constant multi-dimensional view on the instance
  constexpr operator ConstMDSpanType() const { return this->constMDSpan(); }

  //! Conversion to a 1D view on the instance (only if rank == 1)
  constexpr operator SmallSpan<DataType>() requires(Extents::rank() == 1) { return this->to1DSpan().smallView(); }
  //! Conversion to a constant 1D view on the instance (only if rank == 1)
  constexpr operator SmallSpan<const DataType>() const requires(Extents::rank() == 1) { return this->to1DSpan().constSmallView(); }

  //! 1D view on the instance (only if rank == 1)
  constexpr SmallSpan<DataType> to1DSmallSpan() requires(Extents::rank() == 1) { return m_span.to1DSmallSpan(); }
  //! Constant 1D view on the instance (only if rank == 1)
  constexpr SmallSpan<const DataType> to1DSmallSpan() const requires(Extents::rank() == 1) { return m_span.to1DSmallSpan(); }
  //! Constant 1D view on the instance (only if rank == 1)
  constexpr SmallSpan<const DataType> to1DConstSmallSpan() const requires(Extents::rank() == 1) { return m_span.to1DConstSmallSpan(); }

 public:

  //! \internal
  DataType* _internalData() { return m_span._internalData(); }

 private:

  MDSpanType m_span;
  ArrayWrapper m_data;
  Int64 m_total_nb_element = 0;

 private:

  void _updateSpanPointerFromData()
  {
    m_span.m_ptr = m_data.to1DSpan().data();
  }

  void _resizeAndCopy(ConstMDSpanType rhs, eMemoryResource input_ressource, const RunQueue* queue)
  {
    this->resize(rhs.extents().dynamicExtents());
    m_data.copyOnly(rhs.to1DSpan(), input_ressource, queue);
    _updateSpanPointerFromData();
  }

  //! Resizes the array based on the values of \a m_span.extents()
  void _resize()
  {
    m_total_nb_element = m_span.extents().totalNbElement();
    m_data.resize(m_total_nb_element);
    _updateSpanPointerFromData();
  }

  /*!
   * \brief Possible allocation during initialization.
   *
   * An allocation is needed during initialization
   * with the default constructor if all dimensions are static.
   */
  void _resizeInit()
  {
    if constexpr (ExtentsType::nb_dynamic == 0) {
      resize(DynamicDimsType());
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
