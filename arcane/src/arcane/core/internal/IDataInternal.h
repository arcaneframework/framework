// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataInternal.h                                             (C) 2000-2025 */
/*                                                                           */
/* Internal part of IData in Arcane.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
#define ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/IHashAlgorithm.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataCompressor;
class INumericDataInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to manage data compression/decompression.
 */
class DataCompressionBuffer
{
 public:

  UniqueArray<std::byte> m_buffer;
  Int64 m_original_dim1_size = 0;
  Int64 m_original_dim2_size = 0;
  IDataCompressor* m_compressor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for calculating data hash.
 */
class DataHashInfo
{
 public:

  explicit DataHashInfo(IHashAlgorithmContext* context)
  : m_context(context)
  {}

 public:

  IHashAlgorithmContext* context() const { return m_context; }
  Int32 version() const { return m_version; }
  void setVersion(Int32 v) { m_version = v; }

 private:

  IHashAlgorithmContext* m_context = nullptr;
  Int32 m_version = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal part of IData.
 */
class ARCANE_CORE_EXPORT IDataInternal
{
 public:

  virtual ~IDataInternal() = default;

 public:

  /*!
   * \brief Compresses the data and frees the associated memory
   *
   * Compresses the data and fills \a buf with the compressed information.
   * Then it frees the associated memory. The instance will no longer be usable
   * until decompressAndFill() has been called.
   *
   * \retval true if compression occurred.
   * \retval false if the instance does not support compression. In this case
   * it remains usable.
   *
   * \warning Calling this method modifies the underlying container. If
   * this data is associated with a variable, IVariable::syncReferences() must be called.
   */
  virtual bool compressAndClear(DataCompressionBuffer& buf)
  {
    ARCANE_UNUSED(buf);
    return false;
  }

  /*!
   * \brief Decompresses the data and fills the data values.
   *
   * Decompresses the data from \a buf and fills the values of this instance
   * with the decompressed information.
   *
   * \retval true if decompression occurred.
   * \retval false if no decompression occurred because the instance does not support it.
   *
   * \warning Calling this method modifies the underlying container. If
   * this data is associated with a variable, IVariable::syncReferences() must be called.
   */
  virtual bool decompressAndFill(DataCompressionBuffer& buf)
  {
    ARCANE_UNUSED(buf);
    return false;
  }

  //! Generic interface for numeric data (nullptr if the data is not numeric)
  virtual INumericDataInternal* numericData() { return nullptr; }

  /*!
   * \brief Calculates the hash of the data.
   *
   * Outputs the version and value into hash_info.m_version and hash_info.m_value.
   */
  virtual void computeHash(DataHashInfo& hash_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for an 'IData' of a numeric type.
 *
 * Numeric types are the types of eBasicDataType.
 *
 * Generally, all IData are of this type except StringScalarData or
 * StringArrayData.
 */
class ARCANE_CORE_EXPORT INumericDataInternal
{
 public:

  virtual ~INumericDataInternal() = default;

 public:

  //! Memory view of the data
  virtual MutableMemoryView memoryView() = 0;

  //! Number of elements in the first dimension
  virtual Int32 extent0() const = 0;

  /*!
   * \brief Changes the variable's allocator.
   * \warning For experimental use only.
   */
  virtual void changeAllocator(const MemoryAllocationOptions& alloc_info) = 0;

  /*!
   * \brief Allocator used for the data.
   */
  virtual IMemoryAllocator* memoryAllocator() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullDataInternal
: public IDataInternal
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for an array data of type \a T
 */
template <class DataType>
class IArrayDataInternalT
: public IDataInternal
{
 public:

  //! Reserves memory for \a new_capacity elements
  virtual void reserve(Integer new_capacity) = 0;

  //! Container associated with the data.
  virtual Array<DataType>& _internalDeprecatedValue() = 0;

  //! Capacity allocated by the container
  virtual Integer capacity() const = 0;

  //! Frees additional allocated memory
  virtual void shrink() const = 0;

  //! Resizes the container.
  virtual void resize(Integer new_size) = 0;

  //! Clears the container and frees allocated memory.
  virtual void dispose() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a two-dimensional array data of type \a T
 */
template <class DataType>
class IArray2DataInternalT
: public IDataInternal
{
 public:

  //! Reserves memory for \a new_capacity elements
  virtual void reserve(Integer new_capacity) = 0;

  //! Container associated with the data.
  virtual Array2<DataType>& _internalDeprecatedValue() = 0;

  //! Resizes the container only in dimension 1.
  virtual void resizeOnlyDim1(Int32 new_dim1_size) = 0;

  //! Resizes the container.
  virtual void resize(Int32 new_dim1_size, Int32 new_dim2_size) = 0;

  //! Frees additional allocated memory
  virtual void shrink() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*!
 * \brief Copies \a source to \a destination.
 *
 * The memory region \a source must already have the same size as that of
 * the data \a destination.
 */
extern "C++" ARCANE_CORE_EXPORT void
copyContiguousData(INumericDataInternal* destination, ConstMemoryView source, RunQueue& queue);

/*!
 * \brief Copies \a source to \a destination.
 *
 * The data must be of type \a INumericData and the memory region
 * of the destination must already have been allocated to the correct size.
 */
extern "C++" ARCANE_CORE_EXPORT void
copyContiguousData(IData* destination, IData* source, RunQueue& queue);

extern "C++" ARCANE_CORE_EXPORT void
fillContiguousDataGeneric(IData* data, const void* fill_address,
                          Int32 datatype_size, RunQueue& queue);

template <typename DataType> inline void
fillContiguousData(IData* data, const DataType& value, RunQueue& queue)
{
  constexpr Int32 type_size = static_cast<Int32>(sizeof(DataType));
  fillContiguousDataGeneric(data, &value, type_size, queue);
}

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
