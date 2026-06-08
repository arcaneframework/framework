// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Classic 2D array.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_ARRAY2_H
#define ARCCORE_COLLECTIONS_ARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/Span2.h"

#include "arccore/common/Array.h"
#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Class representing a classic 2D array.
 *
 * Instances of this class are neither copyable nor assignable. To create a
 * copyable array, you must use SharedArray2 (for reference semantics) or
 * UniqueArray2 (for value semantics like STL).
 */
template<typename DataType>
class Array2
: private AbstractArray<DataType>
{
 protected:

  enum CloneBehaviour
  {
    CB_Clone,
    CB_Shared
  };
  enum InitBehaviour
  {
    IB_InitWithDefault,
    IB_NoInit
  };

 private:

  using BaseClass = AbstractArray<DataType>;
  typedef AbstractArray<DataType> Base;
  typedef typename Base::ConstReferenceType ConstReferenceType;

 protected:

  using BaseClass::m_ptr;
  using BaseClass::m_md;
  using BaseClass::_setMP2;
  using BaseClass::_setMP;
  using BaseClass::_destroy;
  using BaseClass::_internalDeallocate;
  using BaseClass::_initFromAllocator;
  using BaseClass::_checkValidSharedArray;

 public:

  using AbstractArray<DataType>::allocator;
  using AbstractArray<DataType>::setMemoryLocationHint;
  using AbstractArray<DataType>::changeAllocator;
  using AbstractArrayBase::setDebugName;
  using AbstractArrayBase::debugName;
  using AbstractArrayBase::allocationOptions;

 protected:

  Array2() : AbstractArray<DataType>() {}
  //! Creates an array of \a size1 * \a size2 elements.
  Array2(Int64 size1,Int64 size2)
  : AbstractArray<DataType>() 
  {
    resize(size1,size2);
  }
  Array2(ConstArray2View<DataType> rhs) : AbstractArray<DataType>()
  {
    this->copy(rhs);
  }
  Array2(const Span2<const DataType>& rhs) : AbstractArray<DataType>()
  {
    this->copy(rhs);
  }

 protected:

  //! Creates an empty array with a specific allocator \a allocator
  explicit Array2(IMemoryAllocator* allocator)
  : AbstractArray<DataType>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), 0);
  }
  /*!
   * \brief Creates an array of \a size1 * \a size2 elements with
   * a specific allocator \a allocator.
   */
  Array2(IMemoryAllocator* allocator,Int64 size1,Int64 size2)
  : AbstractArray<DataType>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), size1 * size2);
    resize(size1,size2);
  }
  ~Array2() override = default;

 private:

  //! Deleted
  Array2<DataType>& operator=(const Array2<DataType>& rhs) = delete;
  //! Deleted
  Array2(const Array2<DataType>& rhs) = delete;

 protected:

  //! Move constructor. Only valid for UniqueArray2.
  Array2(Array2<DataType>&& rhs) : AbstractArray<DataType>(std::move(rhs)) {}

 public:

  // TODO: return a Span.
  ArrayView<DataType> operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: return a Span
  ConstArrayView<DataType> operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: return a Span.
  ArrayView<DataType> operator()(Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: return a Span
  ConstArrayView<DataType> operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  DataType& operator()(Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  ConstReferenceType operator()(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  DataType& operator[](Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  ConstReferenceType operator[](Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
#endif
  DataType item(Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  void setItem(Int64 i,Int64 j,ConstReferenceType v)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    m_ptr[ (m_md->dim2_size*i) + j ] = v;
  }
  //! Element at index \a i. Always checks for bounds.
  ConstArrayView<DataType> at(Int64 i) const
  {
    arccoreCheckAt(i,m_md->dim1_size);
    return this->operator[](i);
  }
  //! Element at index \a i. Always checks for bounds.
  ArrayView<DataType> at(Int64 i)
  {
    arccoreCheckAt(i,m_md->dim1_size);
    return this->operator[](i);
  }
  DataType at(Int64 i,Int64 j)
  {
    arccoreCheckAt(i,m_md->dim1_size);
    arccoreCheckAt(j,m_md->dim1_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  void fill(ConstReferenceType v)
  {
    this->_fill(v);
  }
  void clear()
  {
    this->resize(0,0);
  }
  [[deprecated("Y2021: Use SharedArray2::clone() or UniqueArray2::clone()")]]
  Array2<DataType> clone()
  {
    return Array2<DataType>(this->constSpan());
  }
  /*!
   * \brief Resizes the instance based on the dimensions of \a rhs
   * and copies the values of \a rhs into it.
   */
  void copy(Span2<const DataType> rhs)
  {
    _resizeAndCopyView(rhs);
  }
  //! Capacity (number of allocated elements) of the array
  Integer capacity() const { return Base::capacity(); }

  //! Capacity (number of allocated elements) of the array
  Int64 largeCapacity() const { return Base::largeCapacity(); }

  //! Reserves memory for \a new_capacity elements
  void reserve(Int64 new_capacity) { Base::_reserve(new_capacity); }

  // Reallocates memory tightly.
  void shrink() { Base::_shrink(); }

  //! Reallocates memory to have a capacity close to \a new_capacity.
  void shrink(Int64 new_capacity) { Base::_shrink(new_capacity); }

  // Reallocates memory tightly.
  void shrink_to_fit() { Base::_shrink(); }

  //! View of the array as a 1D array
  ArrayView<DataType> viewAsArray()
  {
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->size),m_ptr);
  }
  //! View of the array as a 1D array
  ConstArrayView<DataType> viewAsArray() const
  {
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->size),m_ptr);
  }
  //! View of the array as a 1D array
  Span<DataType> to1DSpan()
  {
    return Span<DataType>(m_ptr,m_md->size);
  }
  //! View of the array as a 1D array
  Span<const DataType> to1DSpan() const
  {
    return Span<const DataType>(m_ptr,m_md->size);
  }

 public:

  operator Array2View<DataType>()
  {
    return view();
  }
  operator ConstArray2View<DataType>() const
  {
    return constView();
  }
  operator Span2<const DataType>() const
  {
    return Span2<const DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  operator Span2<DataType>()
  {
    return Span2<DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  Array2View<DataType> view()
  {
    return Array2View<DataType>(m_ptr,ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size),ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size));
  }
  ConstArray2View<DataType> constView() const
  {
    return ConstArray2View<DataType>(m_ptr,ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size),ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size));
  }
  Span2<DataType> span()
  {
    return Span2<DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  Span2<const DataType> constSpan() const
  {
    return Span2<const DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
 public:

  Integer dim2Size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size); }
  Integer dim1Size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size); }
  Int64 largeDim2Size() const { return m_md->dim2_size; }
  Int64 largeDim1Size() const { return m_md->dim1_size; }
  void add(const DataType& value)
  {
    Base::_addRange(value,m_md->dim2_size);
    ++m_md->dim1_size;
    _arccoreCheckSharedNull();
  }

  /*!
   * \brief Resizes only the first dimension, leaving
   * the second dimension unchanged.
   *
   * Any new values are initialized with the default constructor.
   */
  void resize(Int64 new_size)
  {
    _resize(new_size,IB_InitWithDefault);
  }

  /*!
   * \brief Resizes only the first dimension, leaving
   * the second dimension unchanged.
   *
   * Any new values are NOT initialized.
   */
  void resizeNoInit(Int64 new_size)
  {
    _resize(new_size,IB_NoInit);
  }

  /*!
   * \brief Reallocates both dimensions.
   *
   * Any new values are initialized with the default constructor.
   */
  void resize(Int64 new_size1,Int64 new_size2)
  {
    _resize(new_size1,new_size2,IB_InitWithDefault);
  }

  /*!
   * \brief Reallocates both dimensions.
   *
   * Any new values are NOT initialized.
   */
  void resizeNoInit(Int64 new_size1,Int64 new_size2)
  {
    _resize(new_size1,new_size2,IB_NoInit);
  }

  //! Total number of elements (dim1Size()*dim2Size())
  Integer totalNbElement() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size*m_md->dim2_size); }

  //! Total number of elements (largeDim1Size()*largeDim2Size())
  Int64 largeTotalNbElement() const { return m_md->dim1_size*m_md->dim2_size; }

 protected:

  //! Resizes only the first dimension, leaving the second dimension unchanged
  void _resize(Int64 new_size,InitBehaviour rb)
  {
    Int64 old_size = m_md->dim1_size;
    if (new_size == old_size && !m_md->is_collective_allocator)
      return;
    _resize2(new_size,m_md->dim2_size,rb);
    m_md->dim1_size = new_size;
    _arccoreCheckSharedNull();
  }

  //! Reallocates both dimensions
  void _resize(Int64 new_size1,Int64 new_size2,InitBehaviour rb)
  {
    if (new_size2==m_md->dim2_size){
      _resize(new_size1,rb);
    }
    else if (totalNbElement()==0){
      _resizeFromEmpty(new_size1,new_size2,rb);
    }
    else if (new_size2<m_md->dim2_size){
      _resizeSameDim1ReduceDim2(new_size2,rb);
      _resize(new_size1,rb);
    }
    else if (new_size2>m_md->dim2_size){
      _resizeSameDim1IncreaseDim2(new_size2,rb);
      _resize(new_size1,rb);
    }
    else
      throw NotImplementedException("Array2::resize","already sized");
  }

  void _resizeFromEmpty(Int64 new_size1,Int64 new_size2,InitBehaviour rb)
  {
    _resize2(new_size1,new_size2,rb);
    m_md->dim1_size = new_size1;
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
  }

  void _resizeSameDim1ReduceDim2(Int64 new_size2,InitBehaviour rb)
  {
    ARCCORE_ASSERT((new_size2<m_md->dim2_size),("Bad Size"));
    Int64 n = m_md->dim1_size;
    Int64 n2 = m_md->dim2_size;
    for( Int64 i=0; i<n; ++i ){
      for( Int64 j=0; j<new_size2; ++j )
        m_ptr[(i*new_size2)+j] = m_ptr[(i*n2)+j];
    }
    _resize2(n,new_size2,rb);
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
  }

  void _resizeSameDim1IncreaseDim2(Int64 new_size2,InitBehaviour rb)
  {
    ARCCORE_ASSERT((new_size2>m_md->dim2_size),("Bad Size"));
    Int64 n = m_md->dim1_size;
    Int64 n2 = m_md->dim2_size;
    _resize2(n,new_size2,rb);
    // Recopie en partant de la fin pour éviter tout écrasement
    for( Int64 i=(n-1); i>=0; --i ){
      for( Int64 j=(n2-1); j>=0; --j )
        m_ptr[(i*new_size2)+j] = m_ptr[(i*n2)+j];
    }
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
    if (rb==IB_InitWithDefault){
      // Sets default values for new elements
      for( Int64 i=0; i<n; ++i ){
        for( Int64 j=n2; j<new_size2; ++j ){
          m_ptr[(i*new_size2)+j] = DataType{};
        }
      }
      
    }
  }

  void _resize2(Int64 d1,Int64 d2,InitBehaviour rb)
  {
    Int64 new_size = d1*d2;
    // If the new size is zero, we still need to perform an allocation
    // to store the dim1_size and dim2_size values (otherwise, they would be
    // in TrueImpl::shared_null
    if (new_size==0)
      this->_reserve(4);
    if (rb==IB_InitWithDefault)
      Base::_resize(new_size,DataType());
    else if (rb==IB_NoInit)
      Base::_resize(new_size);
    else
      throw NotSupportedException("Array2::_resize2","invalid value InitBehaviour");
  }

  void _move(Array2<DataType>& rhs)
  {
    Base::_move(rhs);
  }

  void _swap(Array2<DataType>& rhs)
  {
    Base::_swap(rhs);
  }

  // Only valid for UniqueArray2
  void _assignFromArray2(const Array2<DataType>& rhs)
  {
    if (&rhs==this)
      return;
    if (rhs.allocator()==this->allocator()){
      this->copy(rhs.constSpan());
    }
    else{
      this->_assignFromArray(rhs);
      m_md->dim1_size = rhs.dim1Size();
      m_md->dim2_size = rhs.dim2Size();
    }
  }

  void _resizeAndCopyView(Span2<const DataType> rhs)
  {
    Int64 total = rhs.totalNbElement();
    if (total==0){
      // If the size is zero, we still need to perform an allocation
      // to store the dim1_size and dim2_size values (otherwise, they would be
      // in TrueImpl::shared_null)
      this->_reserve(4);
    }
    Span<const DataType> aview(rhs.data(),total);
    Base::_resizeAndCopyView(aview);
    m_md->dim1_size = rhs.dim1Size();
    m_md->dim2_size = rhs.dim2Size();
    _arccoreCheckSharedNull();
  }

 private:

  void _arccoreCheckSharedNull()
  {
    if (!m_ptr)
      ArrayMetaData::throwNullExpected();
    if (!m_md->is_not_null)
      ArrayMetaData::throwNotNullExpected();
  }

 protected:

  void _copyMetaData(const Array2<DataType>& rhs)
  {
    AbstractArray<DataType>::_copyMetaData(rhs);
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief Shared 2D data vector with reference semantics.
 *
 * \code
 * SharedArray2<int> a1(5,7);
 * SharedArray2<int> a2;
 * a2 = a1;
 * a1[3][6] = 1;
 * a2[1][2] = 2;
 * \endcode
 *
 * In the previous example, a1 and a2 refer to the same memory area,
 * so a1[3][6] will have the same value as a2[1][2].
 *
 * To have a vector that copies elements upon assignment,
 * you must use the UniqueArray2 class.
 *
 * For more information, refer to SharedArray.
 */
template<typename T>
class SharedArray2
: public Array2<T>
{
 protected:

  using Array2<T>::m_ptr;
  using Array2<T>::m_md;

 public:

  typedef SharedArray2<T> ThatClassType;
  typedef AbstractArray<T> BaseClassType;
  typedef typename BaseClassType::ConstReferenceType ConstReferenceType;

 public:

  //! Creates an empty array
  SharedArray2() = default;
  //! Creates an array of size1 * size2 elements.
  SharedArray2(Int64 size1,Int64 size2)
  {
    this->resize(size1,size2);
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values from the view.
  SharedArray2(const ConstArray2View<T>& view)
  {
    this->copy(view);
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values from the view.
  SharedArray2(const Span2<const T>& view)
  {
    this->copy(view);
    this->_checkValidSharedArray();
  }
  //! Creates an array referencing rhs.
  SharedArray2(const SharedArray2<T>& rhs)
  : Array2<T>()
  {
    _initReference(rhs);
    this->_checkValidSharedArray();
  }
  //! Creates an array by copying the values of rhs.
  inline SharedArray2(const UniqueArray2<T>& rhs);
  //! Changes the reference of this instance to that of rhs.
  void operator=(const SharedArray2<T>& rhs)
  {
    this->_operatorEqual(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of rhs into this instance.
  inline void operator=(const UniqueArray2<T>& rhs);
  //! Copies the values of the view rhs into this instance.
  void operator=(const ConstArray2View<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copies the values of the view rhs into this instance.
  void operator=(const Span2<const T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Destroys the instance
  ~SharedArray2() override
  {
    _removeReference();
  }
 public:

  //! Clones the array
  SharedArray2<T> clone() const
  {
    return SharedArray2<T>(this->constSpan());
  }

 protected:

  void _initReference(const ThatClassType& rhs)
  {
    this->_setMP(rhs.m_ptr);
    this->_copyMetaData(rhs);
    _addReference(&rhs);
    ++m_md->nb_ref;
  }
  //! Updates references
  void _updateReferences() override final
  {
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      i->_setMP2(m_ptr,m_md);
    for( ThatClassType* i = m_next; i; i = i->m_next )
      i->_setMP2(m_ptr,m_md);
  }
  //! Updates references
  Integer _getNbRef() override final
  {
    Integer nb_ref = 1;
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      ++nb_ref;
    for( ThatClassType* i = m_next; i; i = i->m_next )
      ++nb_ref;
    return nb_ref;
  }
  bool _isUseOwnMetaData() const final
  {
    return false;
  }
  /*!
   * \brief Inserts this instance into the linked list.
   * The instance is inserted at the position of new_ref.
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
  //! Removes this instance from the reference linked list
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
    if (m_md->nb_ref==0){
      this->_destroy();
      this->_internalDeallocate();
    }
  }
  void _operatorEqual(const ThatClassType& rhs)
  {
    if (&rhs!=this){
      _removeReference();
      _addReference(&rhs);
      ++rhs.m_md->nb_ref;
      --m_md->nb_ref;
      _checkFreeMemory();
      this->_setMP2(rhs.m_ptr,rhs.m_md);
    }
  }
 private:

  ThatClassType* m_next = nullptr; //!< Next reference in the linked list
  ThatClassType* m_prev = nullptr; //!< Previous reference in the linked list

 private:

  //! Forbidden
  void operator=(const Array2<T>& rhs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief 2D data vector with value semantics (STL style).
 *
 * This class is the counterpart of UniqueArray for 2D arrays.
 */
template<typename T>
class UniqueArray2
: public Array2<T>
{
 public:

  typedef AbstractArray<T> BaseClassType;
  typedef typename BaseClassType::ConstReferenceType ConstReferenceType;

 public:

 public:
  //! Creates an empty array
  UniqueArray2() : Array2<T>() {}
  //! Creates an array of size1 * size2 elements.
  explicit UniqueArray2(Int64 size1,Int64 size2) : Array2<T>()
  {
    this->resize(size1,size2);
  }
  //! Creates an array by copying the values from the view.
  UniqueArray2(const Span2<const T>& view) : Array2<T>(view) {}
  //! Creates an array by copying the values from the view.
  UniqueArray2(const ConstArray2View<T>& view) : Array2<T>(view) {}
  //! Creates an array by copying the values of rhs.
  UniqueArray2(const Array2<T>& rhs)
  : Array2<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(rhs.allocator()), 0);
    this->_resizeAndCopyView(rhs);
  }
  //! Creates an array by copying the values of rhs.
  UniqueArray2(const UniqueArray2<T>& rhs)
  : Array2<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(rhs.allocator()), 0);
    this->_resizeAndCopyView(rhs);
  }
  //! Creates an array by copying the values of rhs.
  UniqueArray2(const SharedArray2<T>& rhs) : Array2<T>(rhs.constSpan()) {}
  //! Creates an empty array with a specific allocator allocator
  explicit UniqueArray2(IMemoryAllocator* allocator)
  : Array2<T>(allocator) {}
  /*!
   * \brief Creates an array of size1 * size2 elements with
   * a specific allocator allocator.
   */
  UniqueArray2(IMemoryAllocator* allocator,Int64 size1,Int64 size2)
  : Array2<T>(allocator,size1,size2) { }
  //! Move constructor. rhs is invalidated after this call
  UniqueArray2(UniqueArray2<T>&& rhs) ARCCORE_NOEXCEPT : Array2<T>(std::move(rhs)) {}
  //! Copies the values of rhs into this instance.
  UniqueArray2& operator=(const Array2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copies the values of rhs into this instance.
  UniqueArray2& operator=(const SharedArray2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copies the values of rhs into this instance.
  UniqueArray2& operator=(const UniqueArray2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copies the values of the view rhs into this instance.
  UniqueArray2& operator=(ConstArray2View<T> rhs)
  {
    this->copy(rhs);
    return (*this);
  }
  //! Copies the values of the view rhs into this instance.
  UniqueArray2& operator=(const Span2<const T>& rhs)
  {
    this->copy(rhs);
    return (*this);
  }
  //! Move assignment operator. rhs is invalidated after this call.
  UniqueArray2& operator=(UniqueArray2<T>&& rhs) ARCCORE_NOEXCEPT
  {
    this->_move(rhs);
    return (*this);
  }
  //! Destroys the array
  ~UniqueArray2() override = default;
 public:
  /*!
   * \brief Swaps the values of v1 and v2.
   *
   * The swap is done in constant time and without reallocation.
   */
  void swap(UniqueArray2<T>& rhs) ARCCORE_NOEXCEPT
  {
    this->_swap(rhs);
  }
  //! Clones the array
  UniqueArray2<T> clone()
  {
    return UniqueArray2<T>(this->constSpan());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Swaps the values of v1 and v2.
 *
 * The swap is done in constant time and without reallocation.
 */
template<typename T> inline void
swap(UniqueArray2<T>& v1,UniqueArray2<T>& v2)
{
  v1.swap(v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline SharedArray2<T>::
SharedArray2(const UniqueArray2<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline void SharedArray2<T>::
operator=(const UniqueArray2<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
