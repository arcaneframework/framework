// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeInfo.h                                       (C) 2000-2025 */
/*                                                                           */
/* Information for synchronizing data.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZERINFO_H
#define ARCANE_IMPL_DATASYNCHRONIZERINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class DataSynchronizeInfo;

//! Comparison of ghost entity values before/after synchronization
enum class eDataSynchronizeCompareStatus
{
  //! No comparison or unknown result
  Unknown,
  //! Same values before and after synchronization
  Same,
  //! Different values before and after synchronization
  Different
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about the result of a synchronization.
 */
class DataSynchronizeResult
{
 public:

  eDataSynchronizeCompareStatus compareStatus() const { return m_compare_status; }
  void setCompareStatus(eDataSynchronizeCompareStatus v) { m_compare_status = v; }

 private:

  eDataSynchronizeCompareStatus m_compare_status = eDataSynchronizeCompareStatus::Unknown;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about the list of shared/ghost entities for
 * a given rank for a synchronization.
 *
 * TODO: Use a single array for all VariableSyncInfo for shared entities and
 * a single array for ghost entities,
 *       which will be managed by ItemGroupSynchronizeInfo.
 */
class ARCANE_IMPL_EXPORT VariableSyncInfo
{
 public:

  VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids, Int32 rank);
  VariableSyncInfo(const VariableSyncInfo& rhs);
  VariableSyncInfo();

 public:

  //! Target processor rank
  Int32 targetRank() const { return m_target_rank; }

  //! localIds() of entities to send to rank targetRank()
  ConstArrayView<Int32> shareIds() const { return m_share_ids; }
  //! localIds() of entities to receive from rank targetRank()
  ConstArrayView<Int32> ghostIds() const { return m_ghost_ids; }

  //! Number of shared entities
  Int32 nbShare() const { return m_share_ids.size(); }
  //! Number of ghost entities
  Int32 nbGhost() const { return m_ghost_ids.size(); }

  //! Updates the information when the localId() of entities changes
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

 private:

  //! localIds() of entities to send to processor #m_rank
  UniqueArray<Int32> m_share_ids;
  //! localIds() of entities to receive from processor #m_rank
  UniqueArray<Int32> m_ghost_ids;
  //! Target processor rank
  Int32 m_target_rank = A_NULL_RANK;

 private:

  void _changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for sending (share) or receiving (ghost) messages
 */
class DataSynchronizeBufferInfoList
{
  friend DataSynchronizeInfo;

 private:

  DataSynchronizeBufferInfoList(const DataSynchronizeInfo* sync_info, bool is_share)
  : m_sync_info(sync_info)
  , m_is_share(is_share)
  {
  }

 public:

  Int32 nbRank() const { return m_displacements_base.size(); }
  //! Total number of items
  Int64 totalNbItem() const { return m_total_nb_item; }
  //! Displacement in the buffer for rank \a index
  Int64 bufferDisplacement(Int32 index) const { return m_displacements_base[index]; }
  //! Local IDs of entities for rank \a index
  ConstArrayView<Int32> localIds(Int32 index) const;
  //! Number of entities for rank \a index
  Int32 nbItem(Int32 index) const;

 private:

  /*!
   * \brief Offsets in the global buffer for each rank.
   *
   * This array is filled by DataSynchronizeInfo::recompute().
   */
  UniqueArray<Int64> m_displacements_base;
  Int64 m_total_nb_item = 0;
  const DataSynchronizeInfo* m_sync_info = nullptr;
  //! If true, it is the send buffer, otherwise it is the receive buffer.
  bool m_is_share = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information necessary to synchronize entities across a group.
 *
 * recompute() must be called after adding or modifying the instances
 * of VariableSyncInfo.
 *
 * Instances of this class are shared with all dispatchers
 * (IVariableSynchronizeDispatcher) created from an instance of
 * IVariableSynchronizer. Only the latter can therefore modify an instance
 * of this class.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeInfo
: private ReferenceCounterImpl
{
  friend class DataSynchronizeBufferInfoList;

 private:

  static constexpr int SEND = 0;
  static constexpr int RECEIVE = 1;

 private:

  DataSynchronizeInfo() = default;

 public:

  DataSynchronizeInfo(const DataSynchronizeInfo&) = delete;
  DataSynchronizeInfo operator=(const DataSynchronizeInfo&) = delete;
  DataSynchronizeInfo(DataSynchronizeInfo&&) = delete;
  DataSynchronizeInfo operator=(DataSynchronizeInfo&&) = delete;

 public:

  static Ref<DataSynchronizeInfo> create()
  {
    return makeRef(new DataSynchronizeInfo());
  }

 public:

  void clear()
  {
    m_ranks_info.clear();
    m_communicating_ranks.clear();
  }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s);

  //! Send (shared) information
  const DataSynchronizeBufferInfoList& sendInfo() const { return m_buffer_infos[SEND]; }
  //! Receive (ghost) information
  const DataSynchronizeBufferInfoList& receiveInfo() const { return m_buffer_infos[RECEIVE]; }

  //! Rank of the \a index-th target
  Int32 targetRank(Int32 index) const { return m_ranks_info[index].targetRank(); }

  //! Ranks of all targets
  ConstArrayView<Int32> communicatingRanks() const { return m_communicating_ranks; }

  //! Notifies the instance that the local IDs have changed
  void changeLocalIds(ConstArrayView<Int32> old_to_new_ids);

  //! Notifies the instance that the values have changed
  void recompute();

 public:

  void addReference() { ReferenceCounterImpl::addReference(); }
  void removeReference() { ReferenceCounterImpl::removeReference(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  VariableSyncInfo& rankInfo(Int32 i) { return m_ranks_info[i]; }
  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  const VariableSyncInfo& rankInfo(Int32 i) const { return m_ranks_info[i]; }

 private:

  UniqueArray<Int32> m_communicating_ranks;
  UniqueArray<VariableSyncInfo> m_ranks_info;
  std::array<DataSynchronizeBufferInfoList, 2> m_buffer_infos = { { { this, true }, { this, false } } };

 private:

  DataSynchronizeBufferInfoList& _sendInfo() { return m_buffer_infos[SEND]; }
  DataSynchronizeBufferInfoList& _receiveInfo() { return m_buffer_infos[RECEIVE]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
