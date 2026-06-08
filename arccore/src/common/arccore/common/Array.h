// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.h                                                     (C) 2000-2025 */
/*                                                                           */
/* 1D Array.                                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAY_H
#define ARCCORE_COMMON_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/AbstractArray.h"

#include <initializer_list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief Base class for 1D data vectors.
 *
 * This class manipulates a 1D vector (array) of data.
 *
 * Instances of this class are neither copyable nor assignable. To create a
 * copyable array, you must use SharedArray (for reference semantics) or
 * UniqueArray (for value semantics like STL).
 */
template <typename T>
class Array
: public AbstractArray<T>
{
 protected:

  using AbstractArray<T>::m_ptr;
  using AbstractArray<T>::m_md;

 public:

  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  using typename BaseClassType::const_iterator;
  using typename BaseClassType::const_pointer;
  using typename BaseClassType::const_reference;
  using typename BaseClassType::const_reverse_iterator;
  using typename BaseClassType::difference_type;
  using typename BaseClassType::iterator;
  using typename BaseClassType::pointer;
  using typename BaseClassType::reference;
  using typename BaseClassType::reverse_iterator;
  using typename BaseClassType::size_type;
  using typename BaseClassType::value_type;

 protected:

  Array() {}

 protected:

  //! Move constructor (only for UniqueArray)
  Array(Array<T>&& rhs) ARCCORE_NOEXCEPT : AbstractArray<T>(std::move(rhs)) {}

 protected:

  void _initFromInitializerList(std::initializer_list<T> alist)
  {
    Int64 nsize = arccoreCheckArraySize(alist.size());
    this->_reserve(nsize);
    for (const auto& x : alist)
      this->add(x);
  }

 private:

  Array(const Array<T>& rhs) = delete;
  void operator=(const Array<T>& rhs) = delete;

 public:

  ~Array()
  {
  }

 public:

  operator ConstArrayView<T>() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ConstArrayView<T>(s, m_ptr);
  }
  operator ArrayView<T>()
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s, m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  operator Span<T>()
  {
    return Span<T>(m_ptr, m_md->size);
  }
  //! Constant view of this array
  ConstArrayView<T> constView() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ConstArrayView<T>(s, m_ptr);
  }
  //! Constant view of this array
  Span<const T> constSpan() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  /*!
   * \brief Sub-view starting from element \a abegin and containing \a asize elements.
   *
   * If \a (abegin + asize) is greater than the array size,
   * the view is truncated to that size, potentially returning an empty view.
   */
  ConstArrayView<T> subConstView(Int64 abegin, Int32 asize) const
  {
    if (abegin >= m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin, asize), m_ptr + abegin };
  }
  //! Mutable view of this array
  ArrayView<T> view() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s, m_ptr);
  }
  //! Immutable view of this array
  Span<const T> span() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  //! Mutable view of this array
  Span<T> span()
  {
    return Span<T>(m_ptr, m_md->size);
  }
  //! Immutable view of this array
  SmallSpan<const T> smallSpan() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return SmallSpan<const T>(m_ptr, s);
  }
  //! Immutable view of this array
  SmallSpan<const T> constSmallSpan() const
  {
    return smallSpan();
  }
  //! Mutable view of this array
  SmallSpan<T> smallSpan()
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return SmallSpan<T>(m_ptr, s);
  }
  /*!
   * \brief Sub-view starting from element \a abegin and containing \a asize elements.
   *
   * If \a (abegin + asize) is greater than the array size,
   * the view is truncated to that size, potentially returning an empty view.
   */
  ArrayView<T> subView(Int64 abegin, Integer asize)
  {
    if (abegin >= m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin, asize), m_ptr + abegin };
  }
  /*!
   * \brief Extracts a sub-array from a list of indices.
   *
   * The result is stored in \a result whose size must be at least
   * equal to the size of \a indexes.
   */
  void sample(ConstArrayView<Integer> indexes, ArrayView<T> result) const
  {
    const Integer result_size = indexes.size();
    [[maybe_unused]] const Int64 my_size = m_md->size;
    for (Integer i = 0; i < result_size; ++i) {
      Int32 index = indexes[i];
      ARCCORE_CHECK_AT(index, my_size);
      result[i] = m_ptr[index];
    }
  }

 public:

  //! Adds element \a val to the end of the array
  void add(ConstReferenceType val)
  {
    if (m_md->size >= m_md->capacity)
      this->_internalRealloc(m_md->size + 1, true);
    new (m_ptr + m_md->size) T(val);
    ++m_md->size;
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(ConstReferenceType val, Int64 n)
  {
    this->_addRange(val, n);
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(ConstArrayView<T> val)
  {
    this->_addRange(val);
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(Span<const T> val)
  {
    this->_addRange(val);
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(ArrayView<T> val)
  {
    this->_addRange(val);
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(Span<T> val)
  {
    this->_addRange(val);
  }
  //! Adds \a n elements of value \a val to the end of the array
  void addRange(const Array<T>& val)
  {
    this->_addRange(val.constSpan());
  }
  /*!
   * \brief Changes the number of elements in the array to \a s.
   *
   * \note If the new array is larger than the old one, the new
   * elements are not initialized if it is a POD type.
   */
  void resize(Int64 s) { this->_resize(s); }
  /*!
   * \brief Changes the number of elements in the array to \a s.
   *
   * If the new array is larger than the old one, the new
   * elements are initialized with the value \a fill_value.
   */
  void resize(Int64 s, ConstReferenceType fill_value)
  {
    this->_resize(s, fill_value);
  }

  /*!
   * \brief Resizes without initializing new values.
   *
   * \warning This can cause undefined behavior if the type
   * \a T is not trivially copyable because the
   * values are not initialized afterwards and the destructor
   * of \a T will be called upon instance destruction.
   */
  void resizeNoInit(Int64 s)
  {
    this->_resizeNoInit(s, nullptr);
  }

  //! Reserves memory for \a new_capacity elements
  void reserve(Int64 new_capacity)
  {
    this->_reserve(new_capacity);
  }
  /*!
   * \brief Reallocates to free unused memory.
   *
   * After this call, capacity() will be equal to size(). If size()
   * is null or very small, it is possible that capacity() is
   * slightly larger.
   */
  void shrink()
  {
    this->_shrink();
  }

  /*!
   * \brief Reallocates memory to have a capacity close to \a new_capacity.
   */
  void shrink(Int64 new_capacity)
  {
    this->_shrink(new_capacity);
  }

  /*!
   * \brief Reallocates to free unused memory.
   *
   * \sa shrink().
   */
  void shrink_to_fit()
  {
    this->_shrink();
  }

  /*!
   * \brief Removes the entity at index \a index.
   *
   * All elements of this array after the removed one are
   * shifted.
   */
  void remove(Int64 index)
  {
    Int64 s = m_md->size;
    ARCCORE_CHECK_AT(index, s);
    for (Int64 i = index; i < (s - 1); ++i)
      m_ptr[i] = m_ptr[i + 1];
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  /*!
   * \brief Removes the last entity from the array.
   */
  void popBack()
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  //! Element at index \a i. Always checks for overflows
  T& at(Int64 i)
  {
    arccoreCheckAt(i, m_md->size);
    return m_ptr[i];
  }
  //! Element at index \a i. Always checks for overflows
  ConstReferenceType at(Int64 i) const
  {
    arccoreCheckAt(i, m_md->size);
    return m_ptr[i];
  }
  //! Sets the element at index \a i. Always checks for overflows
  void setAt(Int64 i, ConstReferenceType value)
  {
    arccoreCheckAt(i, m_md->size);
    m_ptr[i] = value;
  }
  //! Element at index \a i
  ConstReferenceType item(Int64 i) const { return m_ptr[i]; }
  //! Element at index \a i
  void setItem(Int64 i, ConstReferenceType v) { m_ptr[i] = v; }
  //! Element at index \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Element at index \a i
  T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  ConstReferenceType operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Element at index \a i
  T& operator()(Int64 i)
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Last element of the array
  /*! The array must not be empty */
  T& back()
  {
    ARCCORE_CHECK_AT(m_md->size - 1, m_md->size);
    return m_ptr[m_md->size - 1];
  }
  //! Last element of the array (const)
  /*! The array must not be empty */
  ConstReferenceType back() const
  {
    ARCCORE_CHECK_AT(m_md->size - 1, m_md->size);
    return m_ptr[m_md->size - 1];
  }

  //! First element of the array
  /*! The array must not be empty */
  T& front()
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    return m_ptr[0];
  }

  //! First element of the array (const)
  /*! The array must not be empty */
  ConstReferenceType front() const
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    return m_ptr[0];
  }

  //! Removes the elements from the array
  void clear()
  {
    this->_clear();
  }

  //! Fills the array with the value \a value
  void fill(ConstReferenceType value)
  {
    this->_fill(value);
  }

  /*!
   * \brief Copies the values from \a rhs into the instance.
   *
   * The instance is resized so that this->size()==rhs.size().
   */
  void copy(Span<const T> rhs)
  {
    this->_resizeAndCopyView(rhs);
  }

  //! Clones the array
  [[deprecated("Y2021: Use SharedArray::clone() or UniqueArray::clone()")]]
  Array<T> clone() const
  {
    Array<T> x;
    x.copy(this->constSpan());
    return x;
  }

  //! \internal Access to the root of the array without any protection
  const T* unguardedBasePointer() const { return m_ptr; }
  //! \internal Access to the root of the array without any protection
  T* unguardedBasePointer() { return m_ptr; }

  //! Access to the root of the array without any protection
  const T* data() const { return m_ptr; }
  //! \internal Access to the root of the array without any protection
  T* data() { return m_ptr; }

 public:

  //! Iterator over the first element of the array.
  iterator begin() { return iterator(m_ptr); }

  //! Constant iterator over the first element of the array.
  const_iterator begin() const { return const_iterator(m_ptr); }

  //! Iterator over the first element after the end of the array.
  iterator end() { return iterator(m_ptr + m_md->size); }

  //! Constant iterator over the first element after the end of the array.
  const_iterator end() const { return const_iterator(m_ptr + m_md->size); }

  //! Reverse iterator over the first element of the array.
  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

  //! Reverse iterator over the first element of the array.
  const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }

  //! Reverse iterator over the first element after the end of the array.
  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

  //! Reverse iterator over the first element after the end of the array.
  const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }

 public:

  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr, m_ptr + m_md->size);
  }

  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr, m_ptr + m_md->size);
  }

 public:

  //@{ Methods for STL compatibility.
  //! Adds the element \a val to the end of the array
  void push_back(ConstReferenceType val)
  {
    this->add(val);
  }
  //@}

 private:

  //! Method called from totalview debugger
  static int TV_ttf_display_type(const Array<T>* obj);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief 1D vector of data with reference semantics.
 *
 * To have a vector that uses value semantics (like std::vector),
 * you must use the UniqueArray class.
 *
 * Reference semantics work as follows:
 *
 * \code
 * SharedArray<int> a1(5);
 * SharedArray<int> a2;
 * a2 = a1; // a2 and a1 refer to the same memory area.
 * a1[3] = 1;
 * a2[3] = 2;
 * std::cout << a1[3]; // displays '2'
 * \endcode
 *
 * In the previous example, \a a1 and \a a2 refer to the same area
 * of memory and therefore \a a2[3] will have the same value as
 * \a a1[3] (which is the value \a 2),
 *
 * A shared array is deallocated when there are no more
 * references to this array.
 *
 * \warning Referencing/dereferencing operations (assignment operators,
 * copy operators, and destructors) are not thread-safe. Consequently,
 * this type of array must be used with caution in a multi-threaded environment.
 *
 * \sa UniqueArray.
 */
template <typename T>
class SharedArray
: public Array<T>
{
 protected:

  using AbstractArray<T>::m_md;
  using AbstractArray<T>::m_ptr;

 public:

  typedef SharedArray<T> ThatClassType;
  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  //! Creates an empty array
  SharedArray() = default;
  //! Creates an array of \a size elements containing the value \a value.
  SharedArray(Int64 asize, ConstReferenceType value)
  {
    this->_resize(asize, value);
    this->_checkValidSharedArray();
  }
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(long long asize)
  {
    this->_resize(asize);
    this->_checkValidSharedArray();
  }
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(int asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(unsigned long long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(unsigned long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a size elements containing the default value of type T()
  explicit SharedArray(unsigned int asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Creates an array by copying the values from the value \a view.
  SharedArray(const ConstArrayView<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values from the value \a view.
  SharedArray(const Span<const T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values from the value \a view.
  SharedArray(const ArrayView<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values from the value \a view.
  SharedArray(const Span<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(aview);
    this->_checkValidSharedArray();
  }
  SharedArray(std::initializer_list<T> alist)
  : Array<T>()
  {
    this->_initFromInitializerList(alist);
    this->_checkValidSharedArray();
  }
  //! Creates an array referencing \a rhs.
  SharedArray(const SharedArray<T>& rhs)
  : Array<T>()
  {
    _initReference(rhs);
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values of \a rhs.
  inline SharedArray(const UniqueArray<T>& rhs);

  /*!
   * \brief Creates an empty array with a specific allocator \a allocator.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  explicit SharedArray(IMemoryAllocator* allocator)
  : SharedArray(MemoryAllocationOptions(allocator))
  {
  }

  /*!
   * \brief Creates an empty array with a specific allocator \a allocation_options.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  explicit SharedArray(const MemoryAllocationOptions& allocation_options)
  : Array<T>()
  {
    this->_initFromAllocator(allocation_options, 0);
    this->_checkValidSharedArray();
  }

  /*!
   * \brief Creates an array of \a asize elements with a
   * specific allocator \a allocator.
   *
   * If ArrayTraits<T>::IsPODType is TrueType, the elements are not
   * initialized. Otherwise, the default constructor of T is used.
   */
  SharedArray(IMemoryAllocator* allocator, Int64 asize)
  : SharedArray(MemoryAllocationOptions(allocator), asize)
  {
  }

  /*!
   * \brief Creates an array of \a asize elements with a
   * specific allocator \a allocator.
   *
   * If ArrayTraits<T>::IsPODType is TrueType, the elements are not
   * initialized. Otherwise, the default constructor of T is used.
   */
  SharedArray(const MemoryAllocationOptions& allocation_options, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocation_options, asize);
    this->_resize(asize);
    this->_checkValidSharedArray();
  }

  //!Creates an array with the allocator \a allocator by copying the values \a rhs.
  SharedArray(IMemoryAllocator* allocator, Span<const T> rhs)
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), 0);
    this->_initFromSpan(rhs);
    this->_checkValidSharedArray();
  }

  //! Changes the reference of this instance to that of \a rhs.
  void operator=(const SharedArray<T>& rhs)
  {
    this->_operatorEqual(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of \a rhs into this instance.
  inline void operator=(const UniqueArray<T>& rhs);
  //! Copies the values of the view \a rhs into this instance.
  void operator=(const Span<const T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of the view \a rhs into this instance.
  void operator=(const Span<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of the view \a rhs into this instance.
  void operator=(const ConstArrayView<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of the view \a rhs into this instance.
  void operator=(const ArrayView<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  void operator=(std::initializer_list<T> alist)
  {
    this->clear();
    for (const auto& x : alist)
      this->add(x);
    this->_checkValidSharedArray();
  }
  //! Destroys the array
  ~SharedArray() override
  {
    _removeReference();
  }

 public:

  //! Clones the array
  SharedArray<T> clone() const
  {
    return SharedArray<T>(this->allocator(), this->constSpan());
  }

 protected:

  void _initReference(const ThatClassType& rhs)
  {
    // TODO merge with SharedArray2 implementation
    this->_setMP(rhs.m_ptr);
    this->_copyMetaData(rhs);
    _addReference(&rhs);
    ++m_md->nb_ref;
  }
  //! Update references
  void _updateReferences() final
  {
    // TODO merge with SharedArray2 implementation
    for (ThatClassType* i = m_prev; i; i = i->m_prev)
      i->_setMP2(m_ptr, m_md);
    for (ThatClassType* i = m_next; i; i = i->m_next)
      i->_setMP2(m_ptr, m_md);
  }
  //! Update references
  Integer _getNbRef() final
  {
    // NOTE: to be checked, but when this method is called
    // there is always only one reference.
    // TODO merge with SharedArray2 implementation
    Integer nb_ref = 1;
    for (ThatClassType* i = m_prev; i; i = i->m_prev)
      ++nb_ref;
    for (ThatClassType* i = m_next; i; i = i->m_next)
      ++nb_ref;
    return nb_ref;
  }
  bool _isUseOwnMetaData() const final
  {
    return false;
  }
  /*!
   * \brief Inserts this instance into the linked list.
   * The instance is inserted at the position of \a new_ref.
   * \pre m_prev==0
   * \pre m_next==0;
   */
  void _addReference(const ThatClassType* new_ref)
  {
    ThatClassType* nf = const_cast<ThatClassType*>(new_ref);
    ThatClassType* prev = nf->m_prev;
    nf->m_prev = this;
    m_prev = prev;
    m_next = nf;
    if (prev)
      prev->m_next = this;
  }
  //! Removes this instance from the linked list of references
  void _removeReference()
  {
    if (m_prev)
      m_prev->m_next = m_next;
    if (m_next)
      m_next->m_prev = m_prev;
  }
  //! Destroys the instance if no one references it anymore
  void _checkFreeMemory()
  {
    if (m_md->nb_ref == 0) {
      this->_destroy();
      this->_internalDeallocate();
    }
  }
  void _operatorEqual(const ThatClassType& rhs)
  {
    if (&rhs != this) {
      _removeReference();
      _addReference(&rhs);
      ++rhs.m_md->nb_ref;
      --m_md->nb_ref;
      _checkFreeMemory();
      this->_setMP2(rhs.m_ptr, rhs.m_md);
    }
  }

 private:

  ThatClassType* m_next = nullptr; //!< Next reference in the linked list
  ThatClassType* m_prev = nullptr; //!< Previous reference in the linked list

 private:

  //! Forbidden
  void operator=(const Array<T>& rhs) = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief 1D data vector with value semantics (STL style).
 *
 * This class manages a value array in the same way as the
 * stl::vector class in the STL.

 * Value semantics work as follows:
 *
 * \code
 * UniqueArray<int> a1(5);
 * UniqueArray<int> a2;
 * a2 = a1; // a2 becomes a copy of a1.
 * a1[3] = 1;
 * a2[3] = 2;
 * std::cout << a1[3]; // prints '1'
 * \endcode
 *
 * It is possible to specify a specific memory allocator via
 * the constructor UniqueArray(IMemoryAllocator*). In this case, the allocator
 * specified as an argument must remain valid as long as this instance
 * is used.
 *
 * \warning The allocator is transferred to the destination instance during a
 * call to constructors that take an Array, SharedArray, or
 * UniqueArray as an argument. The same applies to the assignment operator and when
 * calling UniqueArray::swap(). If these calls are considered, it
 * must be guaranteed that the allocator remains valid even after transfer. It
 * is therefore preferable in all cases that the specific allocator used
 * remains valid throughout the application's lifetime.
 *
 * \note If the type is a Plain Object Data (POD) type, the data
 * is not initialized upon allocation or reallocation. The template class
 * ArrayTraits allows specifying whether a type is POD based on the value
 * provided by the type ArrayTraits<T>::IsPODType, which can be
 * FalseType or TrueType. The macro ARCCORE_DEFINE_ARRAY_PODTYPE() allows
 * defining such a type.
 * Unless specialized, only C++ base types are POD.
 *
 * \warning If `ArrayTraits<T>::IsPODType` is false, resizing or copying operations
 * always occur on the host. Therefore, the memory returned by the allocator (allocator())
 * must be accessible on the host.
 */
template <typename T>
class UniqueArray
: public Array<T>
{
 public:

  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  //! Creates an empty array
  UniqueArray() {}
  //! Creates an array of \a size elements containing the value \a value.
  UniqueArray(Int64 req_size, ConstReferenceType value)
  {
    this->_resize(req_size, value);
  }
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(long long asize)
  {
    this->_resize(asize);
  }
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(int asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(unsigned long long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(unsigned long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Creates an array of \a asize elements containing the default value of type T()
  explicit UniqueArray(unsigned int asize)
  : UniqueArray(static_cast<long long>(asize))
  {}

  //! Creates an array by copying the values from the value \a aview.
  UniqueArray(const ConstArrayView<T>& aview)
  : UniqueArray(Span<const T>(aview))
  {
  }
  //! Creates an array by copying the values from the value \a aview.
  UniqueArray(const Span<const T>& aview)
  {
    this->_initFromSpan(aview);
  }
  //! Creates an array by copying the values from the value \a aview.
  UniqueArray(const ArrayView<T>& aview)
  : UniqueArray(Span<const T>(aview))
  {
  }
  //! Creates an array by copying the values from the value \a aview.
  UniqueArray(const Span<T>& aview)
  {
    this->_initFromSpan(aview);
  }
  UniqueArray(std::initializer_list<T> alist)
  {
    this->_initFromInitializerList(alist);
  }
  //! Creates an array by copying the values \a rhs.
  UniqueArray(const Array<T>& rhs)
  {
    this->_initFromAllocator(rhs.allocationOptions(), 0);
    this->_initFromSpan(rhs);
  }
  //! Creates an array by copying the values \a rhs.
  UniqueArray(const UniqueArray<T>& rhs)
  : Array<T>{}
  {
    this->_initFromAllocator(rhs.allocationOptions(), 0);
    this->_initFromSpan(rhs);
  }
  //! Creates an array by copying the values \a rhs.
  UniqueArray(const SharedArray<T>& rhs)
  {
    this->_initFromSpan(rhs);
  }
  //! Move constructor. \a rhs is invalidated after this call
  UniqueArray(UniqueArray<T>&& rhs) ARCCORE_NOEXCEPT : Array<T>(std::move(rhs)) {}
  //! Creates an empty array with a specific allocator \a allocator
  explicit UniqueArray(IMemoryAllocator* allocator)
  : Array<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), 0);
  }
  //! Creates an empty array with a specific allocator \a allocator
  explicit UniqueArray(MemoryAllocationOptions allocate_options)
  : Array<T>()
  {
    this->_initFromAllocator(allocate_options, 0);
  }
  /*!
   * \brief Creates an array of \a asize elements with a
   * specific allocator \a allocator.
   *
   * If ArrayTraits<T>::IsPODType is TrueType, the elements are not
   * initialized. Otherwise, the default constructor of T is used.
   *
   * \warning Initialization occurs on the host, so the memory returned
   * by the allocator must be accessible on the host.
   */
  UniqueArray(IMemoryAllocator* allocator, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), asize);
    this->_resize(asize);
  }
  /*!
   * \brief Creates an array of \a asize elements with a
   * specific allocator \a allocator.
   *
   * If ArrayTraits<T>::IsPODType is TrueType, the elements are not
   * initialized. Otherwise, the default constructor of T is used.
   *
   * \warning Initialization occurs on the host, so the memory returned
   * by the allocator must be accessible on the host.
   */
  UniqueArray(MemoryAllocationOptions allocate_options, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocate_options, asize);
    this->_resize(asize);
  }
  /*!
   * \brief Creates an array with the allocator \a allocator by copying
   * the values \a rhs.
   *
   * \warning The copying occurs on the host, so the memory returned
   * by the allocator must be accessible on the host.
   */
  UniqueArray(IMemoryAllocator* allocator, Span<const T> rhs)
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), 0);
    this->_initFromSpan(rhs);
  }
  /*!
   * \brief Creates an array with the allocator \a allocator by copying
   * the values \a rhs.
   *
   * \warning The copying occurs on the host, so the memory returned
   * by the allocator must be accessible on the host.
   */
  UniqueArray(MemoryAllocationOptions allocate_options, Span<const T> rhs)
  {
    this->_initFromAllocator(allocate_options, 0);
    this->_initFromSpan(rhs);
  }

  //! Copies the values of \a rhs into this instance.
  void operator=(const Array<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Copies the values of \a rhs into this instance.
  void operator=(const SharedArray<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Copies the values of \a rhs into this instance.
  void operator=(const UniqueArray<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Move assignment operator. \a rhs is invalidated after this call.
  void operator=(UniqueArray<T>&& rhs) ARCCORE_NOEXCEPT
  {
    this->_move(rhs);
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
  void operator=(const SmallSpan<T>& rhs)
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
  //! Copies the values of the view \a rhs into this instance.
  void operator=(const SmallSpan<const T>& rhs)
  {
    this->copy(rhs);
  }
  //! Copies the values of the view \a alist into this instance.
  void operator=(std::initializer_list<T> alist)
  {
    this->clear();
    for (const auto& x : alist)
      this->add(x);
  }
  //! Destroys the instance.
  ~UniqueArray() override
  {
  }

 public:

  /*!
   * \brief Swaps the values of the instance with those of \a rhs.
   *
   * The swap also includes the associated allocator (allocator())
   * and any debug information.
   *
   * The swap is performed in constant time and without reallocation.
   */
  void swap(UniqueArray<T>& rhs)
  {
    this->_swap(rhs);
  }

  //! Clones the array
  UniqueArray<T> clone() const
  {
    return UniqueArray<T>(*this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Swaps the values of \a v1 and \a v2.
 *
 * The swap is performed in constant time and without reallocation.
 */
template <typename T> inline void
swap(UniqueArray<T>& v1, UniqueArray<T>& v2)
{
  v1.swap(v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline SharedArray<T>::
SharedArray(const UniqueArray<T>& rhs)
: Array<T>()
, m_next(nullptr)
, m_prev(nullptr)
{
  this->_initFromSpan(rhs.constSpan());
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline void SharedArray<T>::
operator=(const UniqueArray<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of an array in the form of non-modifiable bytes 
 *
 * T must be a POD type.
 */
template <typename T> inline Span<const std::byte>
asBytes(const Array<T>& v)
{
  return asBytes(v.constSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of an array in the form of a writable byte array.
 *
 * T must be a POD type.
 */
template <typename T> inline Span<std::byte>
asWritableBytes(Array<T>& v)
{
  return asWritableBytes(v.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
