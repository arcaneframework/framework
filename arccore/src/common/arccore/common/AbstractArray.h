// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractArray.h                                             (C) 2000-2026 */
/*                                                                           */
/* Base class for arrays.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ABSTRACTARRAY_H
#define ARCCORE_COMMON_ABSTRACTARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include "arccore/common/ArrayTraits.h"
#include "arccore/common/ArrayMetaData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal base class for arrays.
 *
 * This class only manages metadata for arrays such as
 * the number of elements or capacity.
 *
 * \a m_md is a pointer containing the array's metadata. If the
 * array is shared (SharedArray, SharedArray2), then this pointer
 * is dynamically allocated and in this case _isUseOwnMetaData() must
 * return \a false. If the array is not shared (UniqueArray or
 * UniqueArray2), then the metadata is kept directly
 * in the array instance to avoid unnecessary allocations
 * and \a m_md then points to \a m_meta_data. In all cases, you
 * must not use \a m_meta_data directly, but always go through
 * \a m_md.
 */
class ARCCORE_COMMON_EXPORT AbstractArrayBase
{
 public:

  AbstractArrayBase()
  {
    m_md = &m_meta_data;
  }
  virtual ~AbstractArrayBase() = default;

 public:

  IMemoryAllocator* allocator() const
  {
    return m_md->allocation_options.allocator();
  }
  MemoryAllocationOptions allocationOptions() const
  {
    return m_md->allocation_options;
  }
  /*!
   * \brief Sets the array name for debug information.
   *
   * This name can be used, for example, for listing displays.
   */
  void setDebugName(const String& name);
  //! Debug name (null if no name specified)
  String debugName() const;

 protected:

  ArrayMetaData* m_md = nullptr;
  ArrayMetaData m_meta_data;

 protected:

  //! Explicit method for a null RunQueue.
  static constexpr RunQueue* _nullRunQueue() { return nullptr; }

  /*!
   * \brief Indicates if \a m_md refers to \a m_meta_data.
   *
   * This is the case for UniqueArray and UniqueArray2 but
   * not for SharedArray and SharedArray2.
   */
  virtual bool _isUseOwnMetaData() const
  {
    return true;
  }

 protected:

  void _swapMetaData(AbstractArrayBase& rhs)
  {
    std::swap(m_md, rhs.m_md);
    std::swap(m_meta_data, rhs.m_meta_data);
    _checkSetUseOwnMetaData();
    rhs._checkSetUseOwnMetaData();
  }

  void _copyMetaData(const AbstractArrayBase& rhs)
  {
    // Move the metadata
    // Note if m_meta_data is used, m_md must be set to point to our own m_meta_data.
    m_meta_data = rhs.m_meta_data;
    m_md = rhs.m_md;
    _checkSetUseOwnMetaData();
  }

  void _allocateMetaData()
  {
#ifdef ARCCORE_CHECK
    if (m_md->is_not_null)
      ArrayMetaData::throwNullExpected();
#endif
    if (_isUseOwnMetaData()) {
      m_meta_data = ArrayMetaData();
      m_md = &m_meta_data;
    }
    else {
      m_md = new ArrayMetaData();
      m_md->is_allocated_by_new = true;
    }
    m_md->is_not_null = true;
  }

  void _deallocateMetaData(ArrayMetaData* md)
  {
    if (md->is_allocated_by_new)
      delete md;
    else
      *md = ArrayMetaData();
  }

  void _checkValidSharedArray()
  {
#ifdef ARCCORE_CHECK
    if (m_md->is_not_null && !m_md->is_allocated_by_new)
      ArrayMetaData::throwInvalidMetaDataForSharedArray();
#endif
  }

 private:

  void _checkSetUseOwnMetaData()
  {
    if (!m_md->is_allocated_by_new)
      m_md = &m_meta_data;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Abstract base class for a vector.
 *
 * This class cannot be used directly. To use a
 * vector, choose the SharedArray or UniqueArray class.
 */
template <typename T>
class AbstractArray
: public AbstractArrayBase
{
 public:

  typedef typename ArrayTraits<T>::ConstReferenceType ConstReferenceType;
  typedef typename ArrayTraits<T>::IsPODType IsPODType;
  typedef AbstractArray<T> ThatClassType;
  using TrueImpl = T;

 public:

  //! Type of the array elements
  typedef T value_type;
  //! Pointer type of an array element
  typedef value_type* pointer;
  //! Constant pointer type of an array element
  typedef const value_type* const_pointer;
  //! Type of the iterator over an array element
  typedef ArrayIterator<pointer> iterator;
  //! Type of the constant iterator over an array element
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type reference of an array element
  typedef value_type& reference;
  //! Type constant reference of an array element
  typedef ConstReferenceType const_reference;
  //! Type indexing the array
  typedef Int64 size_type;
  //! Type of a distance between array element iterators
  typedef ptrdiff_t difference_type;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 protected:

  //! Constructs an empty vector with the default allocator
  AbstractArray()
  {
  }
  //! Move constructor. Should only be used by UniqueArray
  AbstractArray(ThatClassType&& rhs) ARCCORE_NOEXCEPT
  : m_ptr(rhs.m_ptr)
  {
    _copyMetaData(rhs);
    rhs._reset();
  }

  ~AbstractArray() override
  {
    --m_md->nb_ref;
    _checkFreeMemory();
  }

 public:

  AbstractArray(const AbstractArray<T>& rhs) = delete;
  AbstractArray<T>& operator=(const AbstractArray<T>& rhs) = delete;

 protected:

  static constexpr Int64 typeSize() { return static_cast<Int64>(sizeof(T)); }
  AllocatedMemoryInfo _currentMemoryInfo() const
  {
    return AllocatedMemoryInfo(m_ptr, m_md->size * typeSize(), m_md->capacity * typeSize());
  }

 protected:

  /*!
   * \brief Initializes the array with the view \a view.
   *
   * This method must only be called in a derived class constructor.
   */
  void _initFromSpan(const Span<const T>& view)
  {
    Int64 asize = view.size();
    if (asize != 0) {
      _internalAllocate(asize, _nullRunQueue());
      _createRange(0, asize, view.data());
      m_md->size = asize;
    }
  }

  /*!
   * \brief Constructs an empty vector with a specific allocator \a a.
   *
   * If \a acapacity is not null, memory is allocated to
   * hold \a acapacity elements (but the array remains empty).
   *
   * This method must only be called in a derived class constructor
   * and only by classes using UniqueArray semantics.
   */
  void _initFromAllocator(MemoryAllocationOptions o, Int64 acapacity,
                          void* pre_allocated_buffer = nullptr)
  {
    _directFirstAllocateWithAllocator(acapacity, o, pre_allocated_buffer);
  }

 public:

  //! Frees the memory used by the array.
  void dispose()
  {
    _destroy();
    MemoryAllocationOptions options(m_md->allocation_options);
    _internalDeallocate();
    _setToSharedNull();
    // If we have a specific allocator, we must allocate a
    // block to keep this information.
    if (options.allocator() != m_md->_allocator())
      _directFirstAllocateWithAllocator(0, options);
    _updateReferences();
  }

 public:

  operator ConstArrayView<T>() const
  {
    return ConstArrayView<T>(ARCCORE_CAST_SMALL_SIZE(size()), m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  operator SmallSpan<const T>() const
  {
    return SmallSpan<const T>(m_ptr, ARCCORE_CAST_SMALL_SIZE(size()));
  }

 public:

  //! Number of elements in the vector
  Integer size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->size); }
  //! Number of elements in the vector
  Integer length() const { return ARCCORE_CAST_SMALL_SIZE(m_md->size); }
  //! Capacity (number of allocated elements) of the vector
  Integer capacity() const { return ARCCORE_CAST_SMALL_SIZE(m_md->capacity); }
  //! Number of elements in the vector (in 64 bits)
  Int64 largeSize() const { return m_md->size; }
  //! Number of elements in the vector (in 64 bits)
  Int64 largeLength() const { return m_md->size; }
  //! Capacity (number of allocated elements) of the vector (in 64 bits)
  Int64 largeCapacity() const { return m_md->capacity; }
  //! Capacity (number of allocated elements) of the vector
  bool empty() const { return m_md->size == 0; }
  //! True if the array contains the value element \a v
  bool contains(ConstReferenceType v) const
  {
    const T* ptr = m_ptr;
    for (Int64 i = 0, n = m_md->size; i < n; ++i) {
      if (ptr[i] == v)
        return true;
    }
    return false;
  }

 public:

  //! Element at index \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Element at index \a i
  ConstReferenceType operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }

 public:

  //! Modifies the memory location information
  void setMemoryLocationHint(eMemoryLocationHint new_hint)
  {
    m_md->_setMemoryLocationHint(new_hint, m_ptr, sizeof(T));
  }

  /*!
   * \brief Sets the physical location of the memory region.
   *
   * \warning The caller must guarantee consistency between the allocator
   * and the specified memory region.
   */
  void _internalSetHostDeviceMemoryLocation(eHostDeviceMemoryLocation location)
  {
    m_md->m_host_device_memory_location = location;
  }

  //! Sets the physical location of the memory region.
  eHostDeviceMemoryLocation hostDeviceMemoryLocation() const
  {
    return m_md->m_host_device_memory_location;
  }

 public:

  friend bool operator==(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
  {
    return operator==(Span<const T>(rhs), Span<const T>(lhs));
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend bool operator==(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return operator==(Span<const T>(rhs), lhs);
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend bool operator==(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return operator==(rhs, Span<const T>(lhs));
  }

  friend bool operator!=(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const AbstractArray<T>& val)
  {
    o << Span<const T>(val);
    return o;
  }

 private:

  using AbstractArrayBase::m_meta_data;

 protected:

  // NOTE: These two fields are used for the TTF display of totalview.
  // If their order is changed, the corresponding part must be updated
  // in Arcane's totalview displayer.
  T* m_ptr = nullptr;

 protected:

  //! Reserves memory for \a new_capacity elements
  void _reserve(Int64 new_capacity)
  {
    if (new_capacity <= m_md->capacity) {
      // In the case of a collective allocator, we still have to perform a
      // realloc (the allocator must handle the optimization).
      if (m_meta_data.is_collective_allocator) {
        _internalRealloc(m_md->capacity, false);
      }
      return;
    }
    _internalRealloc(new_capacity, false);
  }
  /*!
   * \brief Reallocates the array for a new capacity equal to \a new_capacity.
   *
   * If the new capacity is less than the old one, nothing happens.
   */
  void _internalRealloc(Int64 new_capacity, bool compute_capacity, RunQueue* queue = nullptr)
  {
    // Note: For shared memory, if one of the pointers is nullptr, then
    // it is for all processes.
    if (_isSharedNull()) {
      if (new_capacity != 0 || m_meta_data.is_collective_allocator)
        _internalAllocate(new_capacity, queue);
      return;
    }

    Int64 acapacity = new_capacity;
    if (compute_capacity) {
      acapacity = m_md->capacity;
      //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
      while (new_capacity > acapacity)
        acapacity = (acapacity == 0) ? 4 : (acapacity + 1 + acapacity / 2);
      //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
    }
    // If the new capacity is less than the current one, do nothing
    // (except for a collective allocator).
    if (acapacity <= m_md->capacity) {
      if (m_meta_data.is_collective_allocator) {
        _internalReallocate(m_md->capacity, queue);
      }
      return;
    }
    _internalReallocate(acapacity, queue);
  }

  void _internalReallocate(Int64 new_capacity, RunQueue* queue)
  {
    if constexpr (std::is_trivially_copyable_v<T>) {
      T* old_ptr = m_ptr;
      Int64 old_capacity = m_md->capacity;
      _directReAllocate(new_capacity, queue);
      bool update = (new_capacity < old_capacity) || (m_ptr != old_ptr);
      if (update) {
        _updateReferences();
      }
    }
    else {
      T* old_ptr = m_ptr;
      ArrayMetaData* old_md = m_md;
      AllocatedMemoryInfo old_mem_info = _currentMemoryInfo();
      Int64 old_size = m_md->size;
      _directAllocate(new_capacity, queue);
      if (m_ptr != old_ptr) {
        for (Int64 i = 0; i < old_size; ++i) {
          new (m_ptr + i) T(old_ptr[i]);
          old_ptr[i].~T();
        }
        m_md->nb_ref = old_md->nb_ref;
        m_md->_deallocate(old_mem_info, queue);
        _updateReferences();
      }
    }
  }

  // Frees the memory
  void _internalDeallocate(RunQueue* queue = nullptr)
  {
    // Note: For shared memory, if one of the pointers is nullptr, then
    // it is for all processes.
    if (!_isSharedNull())
      m_md->_deallocate(_currentMemoryInfo(), queue);
    if (m_md->is_not_null)
      _deallocateMetaData(m_md);
  }
  void _internalAllocate(Int64 new_capacity, RunQueue* queue)
  {
    _directAllocate(new_capacity, queue);
    m_md->nb_ref = _getNbRef();
    _updateReferences();
  }

  void _copyFromMemory(const T* source)
  {
    m_md->_copyFromMemory(m_ptr, source, sizeof(T), _nullRunQueue());
  }

 private:

  /*!
   * \brief Performs the first allocation.
   *
   * If \a pre_allocated_buffer is not null, it is used as a buffer
   * for the first allocation. The caller must ensure that
   * this buffer is valid for the requested capacity. \a pre_allocated_buffer
   * is notably used by the SmallArray allocator.
   */
  void _directFirstAllocateWithAllocator(Int64 new_capacity, MemoryAllocationOptions options,
                                         void* pre_allocated_buffer = nullptr)
  {
    IMemoryAllocator* wanted_allocator = options.allocator();
    if (!wanted_allocator) {
      wanted_allocator = ArrayMetaData::_defaultAllocator();
      options.setAllocator(wanted_allocator);
    }
    _allocateMetaData();
    m_md->allocation_options = options;
    if (new_capacity > 0) {
      if (!pre_allocated_buffer)
        _allocateMP(new_capacity, nullptr);
      else
        _setMPCast(pre_allocated_buffer);
    }
    m_md->nb_ref = _getNbRef();
    m_md->size = 0;
    _updateReferences();
  }

  void _directAllocate(Int64 new_capacity, RunQueue* queue)
  {
    if (!m_md->is_not_null)
      _allocateMetaData();
    _allocateMP(new_capacity, queue);
  }

  void _allocateMP(Int64 new_capacity, RunQueue* queue)
  {
    _setMPCast(m_md->_allocate(new_capacity, typeSize(), queue));
  }

  void _directReAllocate(Int64 new_capacity, RunQueue* queue)
  {
    _setMPCast(m_md->_reallocate(_currentMemoryInfo(), new_capacity, typeSize(), queue));
  }

 public:

  void changeAllocator(const MemoryAllocationOptions& options, RunQueue* queue)
  {
    _setMPCast(m_md->_changeAllocator(options, _currentMemoryInfo(), typeSize(), queue));
    _updateReferences();
  }

  void changeAllocator(const MemoryAllocationOptions& options)
  {
    _setMPCast(m_md->_changeAllocator(options, _currentMemoryInfo(), typeSize(), _nullRunQueue()));
    _updateReferences();
  }

 public:

  void printInfos(std::ostream& o)
  {
    o << " Infos: size=" << m_md->size << " capacity=" << m_md->capacity << '\n';
  }

 protected:

  //! Update references
  virtual void _updateReferences()
  {
  }
  //! Update references
  virtual Integer _getNbRef()
  {
    return 1;
  }
  //! Adds n elements of value val to the end of the array
  void _addRange(ConstReferenceType val, Int64 n)
  {
    Int64 s = m_md->size;
    if ((s + n) > m_md->capacity)
      _internalRealloc(s + n, true);
    for (Int64 i = 0; i < n; ++i)
      new (m_ptr + s + i) T(val);
    m_md->size += n;
  }

  //! Adds n elements of value val to the end of the array
  void _addRange(Span<const T> val)
  {
    Int64 n = val.size();
    const T* ptr = val.data();
    Int64 s = m_md->size;
    if ((s + n) > m_md->capacity)
      _internalRealloc(s + n, true);
    _createRange(s, s + n, ptr);
    m_md->size += n;
  }

  //! Destroys the instance if no one references it
  void _checkFreeMemory()
  {
    if (m_md->nb_ref == 0) {
      _destroy();
      _internalDeallocate(_nullRunQueue());
    }
  }
  void _destroy()
  {
    _destroyRange(0, m_md->size, IsPODType());
  }
  void _destroyRange(Int64, Int64, TrueType)
  {
    // Nothing to do for a POD type.
  }
  void _destroyRange(Int64 abegin, Int64 aend, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      m_ptr[i].~T();
  }
  void _createRangeDefault(Int64, Int64, TrueType)
  {
  }
  void _createRangeDefault(Int64 abegin, Int64 aend, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      new (m_ptr + i) T();
  }
  void _createRange(Int64 abegin, Int64 aend, ConstReferenceType value, TrueType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      m_ptr[i] = value;
  }
  void _createRange(Int64 abegin, Int64 aend, ConstReferenceType value, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      new (m_ptr + i) T(value);
  }
  void _createRange(Int64 abegin, Int64 aend, const T* values)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i) {
      new (m_ptr + i) T(*values);
      ++values;
    }
  }
  void _fill(ConstReferenceType value)
  {
    for (Int64 i = 0, n = size(); i < n; ++i)
      m_ptr[i] = value;
  }
  void _clone(const ThatClassType& orig_array)
  {
    Int64 that_size = orig_array.size();
    _internalAllocate(that_size, _nullRunQueue());
    m_md->size = that_size;
    m_md->dim1_size = orig_array.m_md->dim1_size;
    m_md->dim2_size = orig_array.m_md->dim2_size;
    _createRange(0, that_size, orig_array.m_ptr);
  }
  template <typename PodType>
  void _resizeHelper(Int64 s, PodType pod_type, RunQueue* queue)
  {
    if (s < 0)
      s = 0;
    if (s > m_md->size) {
      this->_internalRealloc(s, false, queue);
      this->_createRangeDefault(m_md->size, s, pod_type);
    }
    else {
      this->_destroyRange(s, m_md->size, pod_type);
      if (m_meta_data.is_collective_allocator) {
        this->_internalRealloc(s, false, queue);
      }
    }
    m_md->size = s;
  }
  void _resize(Int64 s)
  {
    _resizeHelper(s, IsPODType(), _nullRunQueue());
  }
  //! Redimensionne sans initialiser les nouvelles valeurs
  void _resizeNoInit(Int64 s, RunQueue* queue = nullptr)
  {
    _resizeHelper(s, TrueType{}, queue);
  }
  void _clear()
  {
    this->_destroyRange(0, m_md->size, IsPODType());
    m_md->size = 0;
  }
  //! Redimensionne et remplit les nouvelles valeurs avec \a value
  void _resize(Int64 s, ConstReferenceType value)
  {
    if (s < 0)
      s = 0;
    if (s > m_md->size) {
      this->_internalRealloc(s, false);
      this->_createRange(m_md->size, s, value, IsPODType());
    }
    else {
      this->_destroyRange(s, m_md->size, IsPODType());
      if (m_meta_data.is_collective_allocator) {
        this->_internalRealloc(s, false);
      }
    }
    m_md->size = s;
  }
  void _copy(const T* rhs_begin, TrueType)
  {
    _copyFromMemory(rhs_begin);
  }
  void _copy(const T* rhs_begin, FalseType)
  {
    for (Int64 i = 0, n = m_md->size; i < n; ++i)
      m_ptr[i] = rhs_begin[i];
  }
  void _copy(const T* rhs_begin)
  {
    _copy(rhs_begin, IsPODType());
  }

  /*!
   * \brief Redimensionne l'instance et recopie les valeurs de \a rhs.
   *
   * Si la taille diminue, les éléments compris entre size() et rhs.size()
   * sont détruits.
   *
   * \post size()==rhs.size()
   */
  void _resizeAndCopyView(Span<const T> rhs)
  {
    const T* rhs_begin = rhs.data();
    Int64 rhs_size = rhs.size();
    const Int64 current_size = m_md->size;
    T* abegin = m_ptr;
    // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
    if (abegin >= rhs_begin && abegin < (rhs_begin + rhs_size))
      ArrayMetaData::overlapError(abegin, m_md->size, rhs_begin, rhs_size);

    if (rhs_size > current_size) {
      this->_internalRealloc(rhs_size, false);
      // Crée les nouveaux éléments
      this->_createRange(m_md->size, rhs_size, rhs_begin + current_size);
      // Copie les éléments déjà existant
      _copy(rhs_begin);
      m_md->size = rhs_size;
    }
    else {
      this->_destroyRange(rhs_size, current_size, IsPODType{});
      m_md->size = rhs_size;
      _copy(rhs_begin);
    }
  }

  /*!
   * \brief Implements the move assignment operator.
   *
   * This call is only valid for UniqueArray type arrays
   * that have only one reference. The info from \a rhs is directly
   * copied to this instance. In return, \a rhs contains an empty array.
   */
  void _move(ThatClassType& rhs) ARCCORE_NOEXCEPT
  {
    if (&rhs == this)
      return;

    // Comme il n'y a qu'une seule référence sur le tableau actuel, on peut
    // directement libérer la mémoire.
    _destroy();
    _internalDeallocate(_nullRunQueue());

    _setMP(rhs.m_ptr);

    _copyMetaData(rhs);

    // Indique que \a rhs est vide.
    rhs._reset();
  }
  /*!
   * \brief Swaps the values of the instance with those of \a rhs.
   *
   * This call is only valid for UniqueArray type arrays
   * and the exchange is done only by swapping pointers. The operation
   * is therefore constant complexity.
   */
  void _swap(ThatClassType& rhs) ARCCORE_NOEXCEPT
  {
    std::swap(m_ptr, rhs.m_ptr);
    _swapMetaData(rhs);
  }

  void _shrink()
  {
    _shrink(size());
  }

  // Reallocates the memory to have a capacity close to \a new_capacity
  void _shrink(Int64 new_capacity)
  {
    if (_isSharedNull())
      return;
    // On n'augmente pas la capacité avec cette méthode
    if (new_capacity > this->capacity())
      return;
    if (new_capacity < 4)
      new_capacity = 4;
    _internalReallocate(new_capacity, _nullRunQueue());
  }

  /*!
   * \brief Resets the array to an empty array.
   * \warning This method is only valid for UniqueArray and not
   * SharedArray.
   */
  void _reset()
  {
    _setToSharedNull();
  }

  constexpr Integer _clampSizeOffet(Int64 offset, Int32 asize) const
  {
    Int64 max_size = m_md->size - offset;
    if (asize > max_size)
      // On est certain de ne pas dépasser 32 bits car on est inférieur à asize.
      asize = static_cast<Integer>(max_size);
    return asize;
  }

  // Uniquement pour UniqueArray et UniqueArray2
  void _assignFromArray(const AbstractArray<T>& rhs)
  {
    if (&rhs == this)
      return;
    Span<const T> rhs_span(rhs);
    if (rhs.allocator() == this->allocator()) {
      _resizeAndCopyView(rhs_span);
    }
    else {
      _destroy();
      _internalDeallocate(_nullRunQueue());
      _reset();
      _initFromAllocator(rhs.allocationOptions(), 0);
      _initFromSpan(rhs_span);
    }
  }

 protected:

  void _setMP(TrueImpl* new_mp)
  {
    m_ptr = new_mp;
  }

  void _setMP2(TrueImpl* new_mp, ArrayMetaData* new_md)
  {
    _setMP(new_mp);
    // Il ne faut garder le nouveau m_md que s'il est alloué
    // sinon on risque d'avoir des références sur des objets temporaires
    m_md = new_md;
    if (!m_md->is_allocated_by_new)
      m_md = &m_meta_data;
  }

  bool _isSharedNull()
  {
    return m_ptr == nullptr;
  }

 private:

  void _setToSharedNull()
  {
    m_ptr = nullptr;
    m_meta_data = ArrayMetaData();
    m_md = &m_meta_data;
  }
  void _setMPCast(void* p)
  {
    _setMP(reinterpret_cast<TrueImpl*>(p));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
