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

#include "BaseDirectMatrixBuilder.h"

#include <iomanip>
#include <limits>
#include <set>

#include <alien/utils/Precomp.h>

#include <algorithm>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef USE_VMAP
#undef KEY_OF
#undef VALUE_OF
#define KEY_OF(i) (i).key()
#define VALUE_OF(i) (i).value()
#else /* USE_VMAP */
#undef KEY_OF
#undef VALUE_OF
#define KEY_OF(i) (i)->first
#define VALUE_OF(i) (i)->second
#endif /* USE_VMAP */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  DirectMatrixBuilder::DirectMatrixBuilder(IMatrix& matrix,
                                           const DirectMatrixOptions::ResetFlag reset_flag,
                                           const DirectMatrixOptions::SymmetricFlag symmetric_flag)
  : m_matrix(matrix)
  , m_matrix_impl(nullptr)
  , m_row_starts()
  , m_cols()
  , m_values()
  , m_row_sizes()
  , m_reset_flag(reset_flag)
  , m_allocated(false)
  , m_finalized(false)
  , m_symmetric_profile(symmetric_flag == DirectMatrixOptions::eSymmetric)
  , m_nproc(0)
  , m_parallel_mng(nullptr)
  , m_trace(nullptr)
  {
    m_matrix.impl()->lock();
    m_matrix_impl = &m_matrix.impl()->get<BackEnd::tag::simplecsr>(true);

    const MatrixDistribution& dist = m_matrix_impl->distribution();

    m_parallel_mng = dist.parallelMng();

    if (!m_parallel_mng) {
      m_nproc = 1;
    }
    else {
      m_nproc = m_parallel_mng->commSize();
    }

    m_local_size = dist.localRowSize();
    m_global_size = dist.globalRowSize();
    m_local_offset = dist.rowOffset();

    m_col_global_size = m_matrix_impl->colSpace().size();

    const bool never_allocated = (m_matrix_impl->getCSRProfile().getNRow() == 0);
    if (m_reset_flag == DirectMatrixOptions::eResetAllocation or never_allocated) {
      m_reset_flag = DirectMatrixOptions::eResetAllocation;
      m_row_sizes.resize(m_local_size, 0);
    }
    else {
      m_row_sizes.resize(m_local_size);
      SimpleCSRInternal::CSRStructInfo& profile =
      m_matrix_impl->internal().getCSRProfile();
      ConstArrayView<Integer> row_starts = profile.getRowOffset();
      for (Integer i = 0; i < m_local_size; ++i) {
        const Integer row_capacity = row_starts[i + 1] - row_starts[i];
        m_row_sizes[i] = row_capacity;
      }
    }
  }

  /*---------------------------------------------------------------------------*/

  DirectMatrixBuilder::~DirectMatrixBuilder()
  {
    if (!m_finalized) {
      finalize();
    }
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::reserve(
  Integer n, const DirectMatrixOptions::ReserveFlag flag)
  {
    ALIEN_ASSERT((!m_allocated), ("Cannot reserve already allocated matrix"));
    m_reset_flag = DirectMatrixOptions::eResetAllocation;

    if (flag == DirectMatrixOptions::eResetReservation) {
      m_row_sizes.fill(n);
    }
    else {
      ALIEN_ASSERT((flag == DirectMatrixOptions::eExtendReservation),
                   ("Unexpected reservation flag"));
      for (Integer i = 0, is = m_row_sizes.size(); i < is; ++i)
        m_row_sizes[i] += n;
    }
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::reserve(
  const ConstArrayView<Integer> indices, Integer n, const ReserveFlag flag)
  {
    ALIEN_ASSERT((!m_allocated), ("Cannot reserve already allocated matrix"));
    m_reset_flag = DirectMatrixOptions::eResetAllocation;

    if (flag == DirectMatrixOptions::eResetReservation) {
      for (Integer i = 0; i < indices.size(); ++i)
        m_row_sizes[indices[i] - m_local_offset] = n;
    }
    else {
      ALIEN_ASSERT((flag == DirectMatrixOptions::eExtendReservation),
                   ("Unexpected reservation flag"));
      for (Integer i = 0; i < indices.size(); ++i)
        m_row_sizes[indices[i] - m_local_offset] += n;
    }
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::allocate()
  {
    _startTimer();
    ALIEN_ASSERT((!m_allocated), ("Cannot allocate already allocated matrix"));

    if (m_reset_flag == DirectMatrixOptions::eResetAllocation) {
      computeProfile(m_row_sizes);
    }
    SimpleCSRInternal::CSRStructInfo& profile = m_matrix_impl->internal().getCSRProfile();

    profile.setSymmetric(m_symmetric_profile);

    m_row_starts = profile.getRowOffset();
    m_cols = profile.getCols();
    m_values = m_matrix_impl->internal().getValues();

    if (m_reset_flag == DirectMatrixOptions::eResetAllocation or m_reset_flag == DirectMatrixOptions::eResetProfile or profile.getColOrdering() != SimpleCSRInternal::CSRStructInfo::eFull) {
      profile.getColOrdering() = SimpleCSRInternal::CSRStructInfo::eUndef;
      m_row_sizes.fill(0);
    }
    else {
      if (m_matrix_impl->isParallel()) {
        // NEED TO SORT COLS BECAUSE OF DichotomyScan
        if (m_reset_flag == DirectMatrixOptions::eResetValues) {
          for (Integer i = 0; i < m_local_size; ++i) {
            const Integer row_capacity = m_row_starts[i + 1] - m_row_starts[i];
            m_row_sizes[i] = row_capacity;
            auto view = ArrayView<Integer>(row_capacity, m_cols.data() + m_row_starts[i]);
            std::sort(view.begin(), view.end());
          }
        }
        else {
          std::set<std::pair<Integer, Real>> entries;
          for (Integer i = 0; i < m_local_size; ++i) {
            const Integer row_capacity = m_row_starts[i + 1] - m_row_starts[i];
            m_row_sizes[i] = row_capacity;
            entries.clear();
            for (Integer k = m_row_starts[i]; k < m_row_starts[i + 1]; ++k) {
              entries.insert(std::make_pair(m_cols[k], m_values[k]));
            }
            Integer k = 0;
            for (auto e = entries.begin(); e != entries.end(); ++e, ++k) {
              m_cols[k] = e->first;
              m_values[k] = e->second;
            }
          }
        }
      }
      else {
        for (Integer i = 0; i < m_local_size; ++i) {
          const Integer row_capacity = m_row_starts[i + 1] - m_row_starts[i];
          m_row_sizes[i] = row_capacity;
        }
      }
    }

    if (m_reset_flag != DirectMatrixOptions::eNoReset)
      m_values.fill(0.);

    m_allocated = true;
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::addData(
  const Integer iIndex, const Integer jIndex, const Real value)
  {
    _startTimer();
    ALIEN_ASSERT((m_allocated), ("Not allocated matrix"));

    // skip dead zone
    if (iIndex == -1 or jIndex == -1)
      return;
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    if (jIndex < -1 or jIndex >= m_col_global_size)
      throw FatalErrorException("column index undefined");
    const Integer row_start = m_row_starts[local_row];
    Integer& row_size = m_row_sizes[local_row];
    Integer row_capacity = m_row_starts[local_row + 1] - row_start;
    Integer hint_pos; // hint insertion position not used
    Real* found_value = intrusive_vmap_insert(jIndex, hint_pos, row_size, row_capacity,
                                              m_cols.unguardedBasePointer() + row_start,
                                              m_values.unguardedBasePointer() + row_start);
    if (found_value)
      *found_value += value;
    else // Manage extra data storage
      m_extras[local_row][jIndex] += value;
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::addData(const Integer iIndex, const Real factor,
                                    ConstArrayView<Integer> jIndexes, ConstArrayView<Real> jValues)
  {
    _startTimer();
    ALIEN_ASSERT((m_allocated), ("Not allocated matrix"));
    ALIEN_ASSERT((jIndexes.size() == jValues.size()),
                 ("Inconsistent sizes: %d vs %d", jIndexes.size(), jValues.size()));

    if (iIndex == -1)
      return; // skip dead zone
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    Integer& row_size = m_row_sizes[local_row];
    Integer row_capacity = m_row_starts[local_row + 1] - row_start;
    ColValueData* local_extras = nullptr;

    for (Integer i = 0, n = jIndexes.size(); i < n; ++i) {
      const Integer jIndex = jIndexes[i];
      if (jIndex == -1)
        continue; // skip dead zone
      if (jIndex < -1 or jIndex >= m_col_global_size)
        throw FatalErrorException("column index undefined");
      Integer hint_pos; // hint insertion position not used
      Real* found_value = intrusive_vmap_insert(jIndex, hint_pos, row_size, row_capacity,
                                                m_cols.unguardedBasePointer() + row_start,
                                                m_values.unguardedBasePointer() + row_start);
      if (found_value)
        *found_value += factor * jValues[i];
      else // Manage extra data storage
      {
        if (local_extras == nullptr)
          local_extras = &m_extras[local_row];
        (*local_extras)[jIndex] += factor * jValues[i];
      }
    }
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::setData(
  const Integer iIndex, const Integer jIndex, const Real value)
  {
    _startTimer();
    ALIEN_ASSERT((m_allocated), ("Not allocated matrix"));

    // skip dead zone
    if (iIndex == -1 or jIndex == -1)
      return;
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    if (jIndex < -1 or jIndex >= m_col_global_size)
      throw FatalErrorException("column index undefined");
    const Integer row_start = m_row_starts[local_row];
    Integer& row_size = m_row_sizes[local_row];
    Integer row_capacity = m_row_starts[local_row + 1] - row_start;
    Integer hint_pos; // hint insertion position not used
    Real* found_value = intrusive_vmap_insert(jIndex, hint_pos, row_size, row_capacity,
                                              m_cols.unguardedBasePointer() + row_start,
                                              m_values.unguardedBasePointer() + row_start);
    if (found_value)
      *found_value = value;
    else // Manage extra data storage
      m_extras[local_row][jIndex] = value;
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::setData(const Integer iIndex, const Real factor,
                                    ConstArrayView<Integer> jIndexes, ConstArrayView<Real> jValues)
  {
    _startTimer();
    ALIEN_ASSERT((m_allocated), ("Not allocated matrix"));
    ALIEN_ASSERT((jIndexes.size() == jValues.size()),
                 ("Inconsistent sizes: %d vs %d", jIndexes.size(), jValues.size()));

    if (iIndex == -1)
      return; // skip dead zone
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    Integer& row_size = m_row_sizes[local_row];
    Integer row_capacity = m_row_starts[local_row + 1] - row_start;
    ColValueData* local_extras = nullptr;

    for (Integer i = 0, n = jIndexes.size(); i < n; ++i) {
      const Integer jIndex = jIndexes[i];
      if (jIndex == -1)
        continue; // skip dead zone
      if (jIndex < -1 or jIndex >= m_col_global_size)
        throw FatalErrorException("column index undefined");
      Integer hint_pos; // hint insertion position not used
      Real* found_value = intrusive_vmap_insert(jIndex, hint_pos, row_size, row_capacity,
                                                m_cols.unguardedBasePointer() + row_start,
                                                m_values.unguardedBasePointer() + row_start);
      if (found_value)
        *found_value = factor * jValues[i];
      else // Manage extra data storage
      {
        if (local_extras == nullptr)
          local_extras = &m_extras[local_row];
        (*local_extras)[jIndex] = factor * jValues[i];
      }
    }
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::finalize()
  {
    if (m_finalized)
      return;
    squeeze();
    m_matrix.impl()->unlock();
    m_finalized = true;
  }

  /*---------------------------------------------------------------------------*/

  class DirectMatrixBuilder::IndexEnumerator
  {
   public:
    class Finder
    {
     public:
      Finder(ConstArrayView<Integer> indexes, const Integer offset)
      {
        for (Integer i = 0, is = indexes.size(); i < is; ++i)
          m_index_set.insert(indexes[i] - offset);
      }
      bool operator()(const Integer index) const
      {
        return m_index_set.find(index) != m_index_set.end();
      }

     private:
      std::set<Integer> m_index_set;
    };

   public:
    IndexEnumerator(ConstArrayView<Integer> indexes, const Integer offset)
    : m_i(0)
    , m_offset(offset)
    , m_indexes(indexes)
    {}
    [[nodiscard]] bool end() const { return m_i >= m_indexes.size(); }
    void operator++() { ++m_i; }
    Integer operator*() const { return m_indexes[m_i] - m_offset; }
    [[nodiscard]] Integer size() const { return m_indexes.size(); }
    [[nodiscard]] Finder finder() const { return Finder(m_indexes, m_offset); }

   private:
    Integer m_i;
    const Integer m_offset;
    const ConstArrayView<Integer> m_indexes;
    std::set<Integer> m_index_set;
  };

  /*---------------------------------------------------------------------------*/

  class DirectMatrixBuilder::FullEnumerator
  {
   public:
    class Finder
    {
     public:
      explicit Finder(const Integer size)
      : m_size(size)
      {}
      bool operator()(const Integer index) const { return index >= 0 and index < m_size; }

     private:
      const Integer m_size;
    };

   public:
    explicit FullEnumerator(const Integer size)
    : m_i(0)
    , m_size(size)
    {}
    [[nodiscard]] bool end() const { return m_i >= m_size; }
    void operator++() { ++m_i; }
    Integer operator*() const { return m_i; }
    [[nodiscard]] Integer size() const { return m_size; }
    [[nodiscard]] Finder finder() const { return Finder(m_size); }

   private:
    Integer m_i;
    const Integer m_size;
  };

  /*---------------------------------------------------------------------------*/

  String DirectMatrixBuilder::stats() const
  {
    std::ostringstream oss;
    _stats(oss, FullEnumerator(m_local_size));
    return String(oss.str());
  }

  /*---------------------------------------------------------------------------*/

  String DirectMatrixBuilder::stats(ConstArrayView<Integer> ids) const
  {
    std::ostringstream oss;
    _stats(oss, IndexEnumerator(ids, m_local_offset));
    return String(oss.str());
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::squeeze()
  {
    bool need_squeeze = false;

    Integer total_size = 0, total_capacity = 0;
    for (Integer i = 0; i < m_local_size; ++i) {
      const Integer row_size = m_row_sizes[i];
      const Integer row_capacity = m_row_starts[i + 1] - m_row_starts[i];
      total_size += row_size;
      total_capacity += row_capacity;
      need_squeeze |= (row_size != row_capacity);
    }
    need_squeeze |= (!m_extras.empty());
    for (auto i = m_extras.begin(); i != m_extras.end(); ++i)
      total_size += i->second.size();

    // Parallel reduction of the decision
    if (m_parallel_mng)
      need_squeeze = Arccore::MessagePassing::mpAllReduce(
      m_parallel_mng, Arccore::MessagePassing::ReduceMax, need_squeeze);

    // Premature return if no need of squeeze
    if (!need_squeeze)
      return;

    UniqueArray<Integer> row_starts(m_local_size + 1);
    UniqueArray<Integer> cols(total_size);
    UniqueArray<Real> values(total_size);

    Integer offset = 0;
    for (Integer i = 0; i < m_local_size; ++i) {
      auto ifinder = m_extras.find(i);
      if (ifinder == m_extras.end()) { // Algo sans fusion avec les extras
        const Integer row_start_orig = m_row_starts[i];
        const Integer row_start = row_starts[i] = offset;
        const Integer row_size = m_row_sizes[i];
        for (Integer j = 0; j < row_size; ++j) {
          cols[row_start + j] = m_cols[row_start_orig + j];
          values[row_start + j] = m_values[row_start_orig + j];
        }
        offset += row_size;
      }
      else { // Algo avec fusion des extras
        const ColValueData& extra_col_value_data = ifinder->second;
        const Integer row_start_orig = m_row_starts[i];
        const Integer row_start = row_starts[i] = offset;
        const Integer row_size = m_row_sizes[i];

        Integer pos = row_start, j = 0;
        ColValueData::const_iterator extra_j_iter = extra_col_value_data.begin();

        while (true) {
          if (j == row_size) { // Copie la partie restante des extras
            while (extra_j_iter != extra_col_value_data.end()) {
              cols[pos] = KEY_OF(extra_j_iter);
              values[pos] = VALUE_OF(extra_j_iter);
              ++pos;
              ++extra_j_iter;
            }
            break;
          }
          if (extra_j_iter == extra_col_value_data.end()) { // Copie la partie restante des originaux
            while (j < row_size) {
              cols[pos] = m_cols[row_start_orig + j];
              values[pos] = m_values[row_start_orig + j];
              ++pos;
              ++j;
            }
            break;
          }

          // Copie en fusionnant les deux listes
          if (m_cols[row_start_orig + j] < KEY_OF(extra_j_iter)) {
            cols[pos] = m_cols[row_start_orig + j];
            values[pos] = m_values[row_start_orig + j];
            ++pos;
            ++j;
          }
          else {
            cols[pos] = KEY_OF(extra_j_iter);
            values[pos] = VALUE_OF(extra_j_iter);
            ++pos;
            ++extra_j_iter;
          }
        }
        offset += row_size + extra_col_value_data.size();
      }
    }
    row_starts[m_local_size] = offset;
    ALIEN_ASSERT((offset == total_size), ("Inconsistent total size"));

    updateProfile(row_starts, cols, values);
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::computeProfile(const ConstArrayView<Integer> sizes)
  {
    UniqueArray<Integer> m_offset;
    m_offset.resize(m_nproc + 1);
    if (m_parallel_mng) {
      Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                           ConstArrayView<Integer>(1, &m_local_offset), m_offset.subView(0, m_nproc));
    }
    m_offset[m_nproc] = m_global_size;

    SimpleCSRInternal::CSRStructInfo& profile = m_matrix_impl->internal().getCSRProfile();
    profile.init(m_local_size);

    m_row_starts = profile.getRowOffset();

    Integer offset = 0;
    for (Integer i = 0; i < m_local_size; ++i) {
      m_row_starts[i] = offset;
      offset += sizes[i];
    }
    m_row_starts[m_local_size] = offset;

    profile.allocate();
    profile.setTimestamp(m_matrix_impl->timestamp() + 1);

    m_cols = profile.getCols();
    profile.getColOrdering() = SimpleCSRInternal::CSRStructInfo::eUndef;

    m_matrix_impl->allocate();
    m_values = m_matrix_impl->internal().getValues();

#ifndef NDEBUG
    m_values.fill(0.);
    m_cols.fill(-1);
#endif /* NDEBUG */
  }

  /*---------------------------------------------------------------------------*/

  void DirectMatrixBuilder::updateProfile(UniqueArray<Integer>& row_starts,
                                          UniqueArray<Integer>& cols, UniqueArray<Real>& values)
  {
    SimpleCSRInternal::CSRStructInfo& profile = m_matrix_impl->internal().getCSRProfile();
    profile.getRowOffset().copy(row_starts);
    m_row_starts = profile.getRowOffset();
    profile.getCols() = cols;
    m_cols = profile.getCols();
    profile.getColOrdering() = SimpleCSRInternal::CSRStructInfo::eFull;

    UniqueArray<Integer> offset;
    offset.resize(m_nproc + 1);
    if (m_parallel_mng) {
      Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                           ConstArrayView<Integer>(1, &m_local_offset), offset.subView(0, m_nproc));
    }
    offset[m_nproc] = m_global_size;

    m_matrix_impl->allocate();
    m_matrix_impl->internal().getValues() = values; // copy values dans Matrix.getValues()

    if (m_nproc > 1) {
      m_matrix_impl->parallelStart(offset, m_parallel_mng, true);
    }
    else
      m_matrix_impl->sequentialStart();
    m_values = m_matrix_impl->internal().getValues();
  }

  /*---------------------------------------------------------------------------*/

  template <typename Enumerator>
  void DirectMatrixBuilder::_stats(std::ostream& o, const Enumerator& e) const
  {
    bool need_squeeze = false;
    Integer total_used_data = 0;
    Integer min_used_data = (m_local_size > 0) ? std::numeric_limits<Integer>::max() : 0,
            max_used_data = 0;
    Integer total_reserved_data = 0;
    Integer min_reserved_data =
            (m_local_size > 0) ? std::numeric_limits<Integer>::max() : 0,
            max_reserved_data = 0;
    for (Enumerator ie = e; !ie.end(); ++ie) {
      const Integer i = *ie;
      const Integer data_capacity = m_row_starts[i + 1] - m_row_starts[i];
      max_reserved_data = std::max(max_reserved_data, data_capacity);
      min_reserved_data = std::min(min_reserved_data, data_capacity);
      total_reserved_data += data_capacity;
      const Integer data_size = m_row_sizes[i];
      min_used_data = std::min(min_used_data, data_size);
      max_used_data = std::max(max_used_data, data_size);
      total_used_data += data_size;
      need_squeeze |= (data_capacity != data_size);
    }
    min_used_data = std::min(min_used_data, max_used_data);
    min_reserved_data = std::min(min_reserved_data, max_reserved_data);

    Integer total_extra_data = 0;
    Integer min_extra_data =
            (!m_extras.empty()) ? std::numeric_limits<Integer>::max() : 0,
            max_extra_data = 0;
    Integer extra_count = 0;
    typename Enumerator::Finder finder = e.finder();
    for (ExtraRows::const_iterator j = m_extras.begin(); j != m_extras.end(); ++j) {
      if (finder(j->first)) {
        ++extra_count;
        const Integer data_size = j->second.size();
        min_extra_data = std::min(min_extra_data, data_size);
        max_extra_data = std::max(max_extra_data, data_size);
        total_extra_data += data_size;
      }
    }
    min_extra_data = std::min(min_extra_data, max_extra_data);
    need_squeeze |= (extra_count > 0);
    //  Parallel reduction of the decision : uniquement pour affichage
    need_squeeze = Arccore::MessagePassing::mpAllReduce(
    m_parallel_mng, Arccore::MessagePassing::ReduceMax, need_squeeze);

    const Integer size = e.size();
    o << "Total used data                = " << total_used_data << " on " << size
      << " rows\n"
      << "Min / Mean / Max used data     = " << min_used_data << std::setprecision(2)
      << " / " << ((size) ? (double(total_used_data) / size) : 0) << " / "
      << max_used_data << "\n"
      << "Total extra data               = " << total_extra_data << " on " << extra_count
      << " rows\n"
      << "Min / Mean / Max extra data    = " << min_extra_data << std::setprecision(2)
      << " / " << ((extra_count) ? (double(total_extra_data) / extra_count) : 0) << " / "
      << max_extra_data << "\n"
      << "Total reserved data            = " << total_reserved_data
      << std::setprecision(3) << "  ("
      << ((total_used_data) ? (100 * double(total_reserved_data) / total_used_data) : 0)
      << "% of used data)\n"
      << "Min / Mean / Max reserved data = " << min_reserved_data << std::setprecision(2)
      << " / " << ((size) ? (double(total_reserved_data) / size) : 0) << " / "
      << max_reserved_data << "\n"
      << "Need squeeze optimization      = " << std::boolalpha << need_squeeze << "\n";
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
