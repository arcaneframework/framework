// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerComputeList.h                           (C) 2000-2024 */
/*                                                                           */
/* Calculation of the list of entities to synchronize.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERCOMPUTELIST_H
#define ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERCOMPUTELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
class VariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculation of the list of entities to synchronize.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerComputeList
: public TraceAccessor
{
  class RankInfo
  {
   public:

    RankInfo() = default;
    explicit RankInfo(Int32 arank)
    : m_rank(arank)
    {}

   public:

    Int32 rank() const { return m_rank; }
    void setRank(Int32 arank) { m_rank = arank; }

    /*!
     * \brief Comparison operator.
     * An instance is considered less than another if
     * its associated subdomain is smaller than that of the other.
     */
    bool operator<(const RankInfo& ar) const
    {
      return m_rank < ar.m_rank;
    }

   private:

    Int32 m_rank = A_NULL_RANK;
  };

  class GhostRankInfo
  : public RankInfo
  {
   public:

    GhostRankInfo() = default;
    explicit GhostRankInfo(Int32 arank)
    : RankInfo(arank)
    , m_nb_item(0)
    {}
    GhostRankInfo(Int32 arank, Integer nb_item)
    : RankInfo(arank)
    , m_nb_item(nb_item)
    {}

   public:

    void setInfos(Int32 arank, SharedArray<Int32>& local_ids)
    {
      setRank(arank);
      m_nb_item = local_ids.size();
      m_local_ids = local_ids;
    }
    Int32ConstArrayView localIds() const { return m_local_ids; }
    Integer nbItem() const { return m_nb_item; }
    void resize() { m_unique_ids.resize(m_nb_item); }
    Int64ArrayView uniqueIds() { return m_unique_ids; }

   private:

    Integer m_nb_item = 0;
    SharedArray<Int32> m_local_ids;
    SharedArray<Int64> m_unique_ids;
  };

  class ShareRankInfo
  : public RankInfo
  {
   public:

    ShareRankInfo() = default;
    ShareRankInfo(Int32 arank, Integer nb_item)
    : RankInfo(arank)
    , m_nb_item(nb_item)
    {}
    explicit ShareRankInfo(Int32 arank)
    : RankInfo(arank)
    {}

   public:

    void setInfos(Int32 arank, SharedArray<Int32>& local_ids)
    {
      setRank(arank);
      m_nb_item = local_ids.size();
      m_local_ids = local_ids;
    }
    Int32ConstArrayView localIds() const { return m_local_ids; }
    void setLocalIds(SharedArray<Int32>& v) { m_local_ids = v; }
    Integer nbItem() const { return m_nb_item; }
    void resize() { m_unique_ids.resize(m_nb_item); }
    Int64ArrayView uniqueIds() { return m_unique_ids; }

   private:

    Integer m_nb_item = 0;
    SharedArray<Int32> m_local_ids;
    SharedArray<Int64> m_unique_ids;
  };

 public:

  explicit VariableSynchronizerComputeList(VariableSynchronizer* var_sync);

 public:

  void compute();

 private:

  VariableSynchronizer* m_synchronizer;
  IParallelMng* m_parallel_mng = nullptr;
  ItemGroup m_item_group;
  bool m_is_verbose = false;
  bool m_is_debug = false;

 private:

  void _createList(UniqueArray<SharedArray<Int32>>& boundary_items);
  void _checkValid(ArrayView<GhostRankInfo> ghost_ranks_info,
                   ArrayView<ShareRankInfo> share_ranks_info);
  void _printSyncList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
