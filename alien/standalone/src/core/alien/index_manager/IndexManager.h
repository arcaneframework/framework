/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <alien/index_manager/IAbstractFamily.h>
#include <alien/index_manager/ScalarIndexSet.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/* TODO: Optimize by vectorizing access for internal structures (getIndex) */
class ALIEN_EXPORT IndexManager
{
 public:
  typedef UniqueArray<ScalarIndexSet> VectorIndexSet;

  typedef std::vector<ScalarIndexSet*> ScalarIndexSetVector;

  enum eKeepAlive
  {
    Clone,
    DontClone
  };

 public:
  class Iterator
  {
   public:
    explicit Iterator(const ScalarIndexSetVector::iterator& it)
    : m_iterator(it)
    {}
    void operator++() { m_iterator++; }
    bool operator!=(const Iterator& it) { return m_iterator != it.m_iterator; }
    ScalarIndexSet& operator*() { return **m_iterator; }

   private:
    ScalarIndexSetVector::iterator m_iterator;
  };

  class ConstIterator
  {
   public:
    explicit ConstIterator(const ScalarIndexSetVector::const_iterator& it)
    : m_iterator(it)
    {}
    void operator++() { m_iterator++; }
    bool operator!=(const ConstIterator& it) { return m_iterator != it.m_iterator; }
    const ScalarIndexSet& operator*() { return **m_iterator; }

   private:
    ScalarIndexSetVector::const_iterator m_iterator;
  };

 public:
  struct InternalEntryIndex
  {
    Integer m_entry_uid;
    Integer m_entry_kind;
    Int64 m_item_uid;
    Integer m_item_localid;
    Integer m_item_index;
    Integer m_item_owner;

    bool operator==(const InternalEntryIndex& m) const
    {
      return m_entry_uid == m.m_entry_uid && m.m_item_localid == m_item_localid;
    }
  };

 public:
  explicit IndexManager(
  Alien::IMessagePassingMng* parallelMng, Alien::ITraceMng* traceMng = nullptr);

  virtual ~IndexManager();

  //! Initialisation
  void init();

  bool isPrepared() const { return m_state == Prepared; }

  void setVerboseMode(bool verbose);

  /*! Prepare
   * Compute all indices, using the specified sorting algorithm.
   */
  template <typename T>
  void prepare(T&& t);

  /*! Prepare
   * Compute all indices, using default algorithm.
   */
  void prepare();

  /* @defgroup stats Index characteristics.
   * Only valid after call to prepare.
   * @{
   */
  void stats(Integer& globalSize, Integer& minLocalIndex, Integer& localSize) const;

  Integer globalSize() const;

  Integer minLocalIndex() const;

  Integer localSize() const;
  /*! }@ */

  /*! @defgroup new Define new entries.
   * @{
   */
  /*! New scalar entry, defined on a set of abstract items.
   *
   * @param name
   * @param localIds
   * @param family
   * @param kind
   * @param alive
   * @return
   */
  ScalarIndexSet buildScalarIndexSet(const String& name,
                                     ConstArrayView<Integer> localIds,
                                     const IAbstractFamily& family,
                                     Integer kind, eKeepAlive alive = DontClone);

  /*! New scalar entry, on elements of a family.
   *
   * @param name
   * @param family
   * @param kind
   * @param alive
   * @return
   */
  ScalarIndexSet buildScalarIndexSet(const String& name, const IAbstractFamily& family,
                                     Integer kind, eKeepAlive alive = DontClone);

  /*! New vector entry, on a set of abstract items.
   *
   * @param name
   * @param localIds
   * @param family
   * @param kind
   * @param alive
   * @return
   *
   * Current implementation handles multi-scalar entries as vector.
   */
  VectorIndexSet buildVectorIndexSet(const String& name,
                                     ConstArrayView<Integer> localIds,
                                     const IAbstractFamily& family,
                                     const UniqueArray<Integer>& kind,
                                     eKeepAlive alive = DontClone);
  /*! New vector entry, on elements of a family.
   *
   * @param name
   * @param family
   * @param kind
   * @param alive
   * @return
   *
   * Current implementation handles multi-scalar entries as vector.
   */
  VectorIndexSet buildVectorIndexSet(const String& name,
                                     const IAbstractFamily& family,
                                     const UniqueArray<Integer>& kind,
                                     eKeepAlive alive = DontClone);
  /*! }@ */

  /*! Remove a entities from the index.
   *
   * @param entry
   * @param localIds
   *
   * Must be called before prepare.
   */
  void removeIndex(const ScalarIndexSet& entry, ConstArrayView<Integer> localIds);

  //! Give a translation table, indexed by items.
  UniqueArray<Integer> getIndexes(const ScalarIndexSet& entry) const;

  //! Give a vector translation table, indexed by items then by entries.
  UniqueArray2<Integer> getIndexes(const VectorIndexSet& entries) const;

  ConstArrayView<Integer> getOwnIndexes(const ScalarIndexSet& entry) const;
  ConstArrayView<Integer> getOwnLocalIds(const ScalarIndexSet& entry) const;
  ConstArrayView<Integer> getAllIndexes(const ScalarIndexSet& entry) const;
  ConstArrayView<Integer> getAllLocalIds(const ScalarIndexSet& entry) const;

  const IAbstractFamily& getFamily(const ScalarIndexSet& entry) const;

  //! Parallel Manager used for the index computation.
  IMessagePassingMng* parallelMng() const { return m_parallel_mng; }

  //! define null index : default = -1, if true null_index = max_index+1
  void setMaxNullIndexOpt(bool flag);

  Integer nullIndex() const;

  Iterator begin() { return Iterator(m_entries.begin()); }
  Iterator end() { return Iterator(m_entries.end()); }
  ConstIterator begin() const { return ConstIterator(m_entries.begin()); }
  ConstIterator end() const { return ConstIterator(m_entries.end()); }

 private:
  ScalarIndexSet buildEntry(
  const String& name, const IAbstractFamily* itemFamily, Integer kind);

  void defineIndex(const ScalarIndexSet& entry, ConstArrayView<Integer> localIds);

  typedef std::vector<InternalEntryIndex> EntryIndexMap;

  void begin_prepare(EntryIndexMap& entry_index);
  void begin_parallel_prepare(EntryIndexMap& entry_index);
  void end_parallel_prepare(EntryIndexMap& entry_index);
  void sequential_prepare(EntryIndexMap& entry_index);
  void end_prepare(EntryIndexMap& entryIndex);

  const IAbstractFamily* addNewAbstractFamily(
  const IAbstractFamily* family, eKeepAlive alive);

 private:
  Alien::IMessagePassingMng* m_parallel_mng = nullptr;
  Alien::ITraceMng* m_trace_mng = nullptr;
  Integer m_local_owner; //!< current owner.

  enum State
  {
    Undef,
    Initialized,
    Prepared
  } m_state;

  bool m_verbose;

  Integer m_local_entry_count;
  Integer m_global_entry_count;
  Integer m_global_entry_offset;
  Integer m_local_removed_entry_count;
  Integer m_global_removed_entry_count;

  bool m_max_null_index_opt;

  //! Table des Entry connues localement
  ScalarIndexSetVector m_entries;

  //! Abstract families and associated clones (if handled)
  std::map<const IAbstractFamily*, std::shared_ptr<IAbstractFamily>> m_abstract_families;

  //! Local ids, sorted by owned then ghosts. By entry.
  std::map<Integer, UniqueArray<Integer>> m_entry_all_items;

  //! Unique ids, sorted by owned then ghosts. By entry.
  std::map<Integer, UniqueArray<Integer>> m_entry_all_indices;

  //! Local ids, only for owned, by entry.
  std::map<Integer, ConstArrayView<Integer>> m_entry_own_items;

  //! Unique ids, only for owned, by entry.
  std::map<Integer, ConstArrayView<Integer>> m_entry_own_indices;

  //! Family, by entry.
  std::map<Integer, const IAbstractFamily*> m_entry_families;

 protected:
  //! \internal Structure interne temporaire dans prepare(), defineEntry() et
  //! removeIndex()
  struct EntryLocalId;
  std::map<Integer, std::shared_ptr<EntryLocalId>> m_entry_local_ids;

  struct ParallelRequests;
  std::shared_ptr<ParallelRequests> parallel;

  //! \internal Structure interne de communication dans prepare()
  struct EntrySendRequest;
  struct EntryRecvRequest;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
void IndexManager::prepare(T&& t)
{
  EntryIndexMap entry_index;

  begin_prepare(entry_index);

  if (m_parallel_mng->commSize() > 1) {
    begin_parallel_prepare(entry_index);

    std::sort(entry_index.begin(), entry_index.end(), t);

    end_parallel_prepare(entry_index);
  }
  else {
    std::sort(entry_index.begin(), entry_index.end(), t);

    sequential_prepare(entry_index);
  }

  end_prepare(entry_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
