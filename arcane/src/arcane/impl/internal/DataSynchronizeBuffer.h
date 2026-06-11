// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeBuffer.h                                     (C) 2000-2025 */
/*                                                                           */
/* Implementation of a generic buffer for data synchronization.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZEBUFFER_H
#define ARCANE_IMPL_DATASYNCHRONIZEBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/utils/FixedArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class IBufferCopier;
class DataSynchronizeResult;
class DataSynchronizeInfo;
class DataSynchronizeBufferInfoList;
class MemoryBuffer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for the IDataSynchronizeBuffer implementation.
 *
 * This implementation uses a single memory buffer to manage the three
 * parts of the synchronization: the send buffer, the receive buffer,
 * and the buffer for comparing if the synchronization modified values
 * (the latter is optional).
 * Each buffer is then divided into N parts, called sub-buffers,
 * where N is the number of ranks communicating. Finally, each sub-buffer is
 * itself divided into P parts, where P is the number of data items to communicate.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeBufferBase
: public IDataSynchronizeBuffer
{
  /*!
   * \brief Buffer for one synchronization element (send, receive, or comparison)
   */
  class BufferInfo
  {
   public:

    //! Global buffer
    MutableMemoryView globalBuffer() const { return m_memory_view; }

    //! Positions the global buffer.
    void setGlobalBuffer(MutableMemoryView v);

    //! Buffer for the \a index-th rank
    MutableMemoryView localBuffer(Int32 rank_index) const;

    //! Buffer for the \a index-th rank and the \a data_index-th data item
    MutableMemoryView dataLocalBuffer(Int32 rank_index, Int32 data_index) const;

    //! Displacement in \a globalBuffer() for the \a index-th rank
    Int64 displacement(Int32 rank_index) const;

    //! Size (in bytes) of the local buffer for rank \a rank_index.
    Int64 localBufferSize(Int32 rank_index) const;

    //! Total size in bytes of the global buffer
    Int64 totalSize() const { return m_total_size; }

    //! Local IDs of entities for rank \a index
    ConstArrayView<Int32> localIds(Int32 index) const;

    void checkValid() const
    {
      ARCANE_CHECK_POINTER(m_buffer_info);
    }

    void initialize(ConstArrayView<Int32> datatype_sizes, const DataSynchronizeBufferInfoList* buffer_info);

   private:

    /*!
     * \brief View onto the memory area of the buffer.
     *
     * This variable is only valid after all buffers have been allocated.
     */
    MutableMemoryView m_memory_view;
    //! Offset (in bytes) in globalBuffer() for each data item
    UniqueArray2<Int64> m_displacements;
    //! Size (in bytes) of each local buffer.
    SmallArray<Int64> m_local_buffer_size;
    //! Size (in bytes) of the type of each data item.
    ConstArrayView<Int32> m_datatype_sizes;
    //! Total size (in bytes) of the buffer
    Int64 m_total_size = 0;
    const DataSynchronizeBufferInfoList* m_buffer_info = nullptr;
  };

 public:

  Int32 nbRank() const final { return m_nb_rank; }
  Int32 targetRank(Int32 index) const final;
  bool hasGlobalBuffer() const final { return true; }

  MutableMemoryView receiveBuffer(Int32 index) final { return m_ghost_buffer_info.localBuffer(index); }
  MutableMemoryView sendBuffer(Int32 index) final { return m_share_buffer_info.localBuffer(index); }

  Int64 receiveDisplacement(Int32 index) const final { return m_ghost_buffer_info.displacement(index); }
  Int64 sendDisplacement(Int32 index) const final { return m_share_buffer_info.displacement(index); }

  MutableMemoryView globalReceiveBuffer() final { return m_ghost_buffer_info.globalBuffer(); }
  MutableMemoryView globalSendBuffer() final { return m_share_buffer_info.globalBuffer(); }

  Int64 totalReceiveSize() const final { return m_ghost_buffer_info.totalSize(); }
  Int64 totalSendSize() const final { return m_share_buffer_info.totalSize(); }

  void barrier() final;

 public:

  DataSynchronizeBufferBase(DataSynchronizeInfo* sync_info, Ref<IBufferCopier> copier);

 public:

  //! Indicates whether values are compared before/after synchronization
  bool isCompareSynchronizedValues() const { return m_is_compare_sync_values; }

  void setSynchronizeBuffer(Ref<MemoryBuffer> v)
  {
    m_memory = v;
  }

  /*!
   * \brief Prepares the synchronization.
   *
   * Prepares the synchronization and allocates buffers if necessary.
   *
   * If \a is_compare_sync is true, the values of ghost entities are compared after synchronization
   * with their value before synchronization.
   *
   * setSynchronizeBuffer() must be called at least once before calling
   * this method to position the allocated memory area.
   */
  virtual void prepareSynchronize(bool is_compare_sync) = 0;

 protected:

  void _allocateBuffers();
  //! Computes the information for the synchronization
  void _compute(ConstArrayView<Int32> datatype_sizes);

 protected:

  DataSynchronizeInfo* m_sync_info = nullptr;
  //! Buffer for all data of ghost entities used for reception
  BufferInfo m_ghost_buffer_info;
  //! Buffer for all data of shared entities used for sending
  BufferInfo m_share_buffer_info;
  //! Buffer for testing if synchronization modified the values of ghost cells
  BufferInfo m_compare_sync_buffer_info;

 protected:

  Int32 m_nb_rank = 0;
  bool m_is_compare_sync_values = false;

  //! Buffer containing the concatenated data for sending and receiving
  Ref<MemoryBuffer> m_memory;

  Ref<IBufferCopier> m_buffer_copier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief IDataSynchronizeBuffer implementation for a single data item
 */
class ARCANE_IMPL_EXPORT SingleDataSynchronizeBuffer
: public TraceAccessor
, public DataSynchronizeBufferBase
{
 public:

  SingleDataSynchronizeBuffer(ITraceMng* tm, DataSynchronizeInfo* sync_info, Ref<IBufferCopier> copier)
  : TraceAccessor(tm)
  , DataSynchronizeBufferBase(sync_info, copier)
  {}

 public:

  void copyReceiveAsync(Int32 index) final;
  void copySendAsync(Int32 index) final;

 public:

  void setDataView(MutableMemoryView v)
  {
    m_data_view = v;
    m_datatype_sizes[0] = v.datatypeSize();
  }
  //! Memory area containing the values of the data to be synchronized
  MutableMemoryView dataView() { return m_data_view; }
  void prepareSynchronize(bool is_compare_sync) override;

  /*!
   * \brief Finalizes the synchronization.
   */
  DataSynchronizeResult finalizeSynchronize();

 private:

  //! View onto the data variable
  MutableMemoryView m_data_view;
  //! Array containing the sizes of the data types
  FixedArray<Int32, 1> m_datatype_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief IDataSynchronizeBuffer implementation for multiple data items.
 */
class ARCANE_IMPL_EXPORT MultiDataSynchronizeBuffer
: public TraceAccessor
, public DataSynchronizeBufferBase
{

 public:

  MultiDataSynchronizeBuffer(ITraceMng* tm, DataSynchronizeInfo* sync_info,
                             Ref<IBufferCopier> copier)
  : TraceAccessor(tm)
  , DataSynchronizeBufferBase(sync_info, copier)
  {}

 public:

  void copyReceiveAsync(Int32 rank_index) final;
  void copySendAsync(Int32 rank_index) final;

 public:

  void setNbData(Int32 nb_data)
  {
    m_data_views.resize(nb_data);
    m_datatype_sizes.resize(nb_data);
  }
  void setDataView(Int32 index, MutableMemoryView v)
  {
    m_data_views[index] = v;
    m_datatype_sizes[index] = v.datatypeSize();
  }

  void prepareSynchronize(bool is_compare_sync) override;

 private:

  //! View onto the data variables
  SmallArray<MutableMemoryView> m_data_views;
  //! Array containing the sizes of the data types
  SmallArray<Int32> m_datatype_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
