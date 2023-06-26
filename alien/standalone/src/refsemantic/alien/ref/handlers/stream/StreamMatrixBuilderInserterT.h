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

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::BaseInserter::BaseInserter()
: m_parent(NULL)
{
  throw FatalErrorException(A_FUNCINFO, "Virtual Ctor never called");
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::BaseInserter::BaseInserter(
StreamMatrixBuilderT<ValueT>* parent, Integer id)
: m_id(id)
, m_index(0)
, m_current_size(0)
, m_current_k(NULL)
, m_values(NULL)
, m_count(0)
, m_size(0)
, m_block_size(1)
, m_parent(parent)
{}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::BaseInserter::~BaseInserter() {}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::BaseInserter::init()
{
  m_index = 0;
  m_current_size = 0;
  m_current_k = NULL;
  m_values = NULL;
  m_count = 0;
  m_size = 0;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamMatrixBuilderT<ValueT>::BaseInserter::getId() const
{
  return m_id;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::BaseInserter::end()
{
  init();
  m_n.resize(0);
  m_row_index.resize(0);
  m_col_index.resize(0);
  m_data_index.resize(0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::BaseInserter::setMatrixValues(
ValueT* matrix_values, Integer block_size)
{
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  m_values = matrix_values;
  m_index = 0;
  m_current_size = m_n[0];
  m_current_k = m_data_index.data();
  m_block_size = block_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::reserve(Integer capacity)
{
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  this->m_row_index.reserve(capacity);
  this->m_col_index.reserve(capacity);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamMatrixBuilderT<ValueT>::BaseInserter::size()
{
  return m_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamMatrixBuilderT<ValueT>::BaseInserter::count()
{
  return m_count;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::start()
{
  ALIEN_ASSERT((this->m_parent->m_state == eStart), ("Inconsistent state"));
  ALIEN_ASSERT((this->m_values != NULL), ("Inserter is not ready for filling"));
  this->m_index = 0;
  this->m_current_size = this->m_n[0];
  this->m_current_k = this->m_data_index.data();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::setEnd()
{
  this->m_index = this->m_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
bool StreamMatrixBuilderT<ValueT>::Filler::isBegin()
{
  return (this->m_index == 0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
bool StreamMatrixBuilderT<ValueT>::Filler::isEnd()
{
  return (this->m_index == this->m_size);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamMatrixBuilderT<ValueT>::Filler::currentSize()
{
  return this->m_current_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamMatrixBuilderT<ValueT>::Filler::index()
{
  return this->m_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntries(
ConstArrayView<Integer> row_index, ConstArrayView<Integer> col_index)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  // ALIEN_ASSERT((row_index.size()==col_index.size()),("row and col with different
  // sizes")) ;
  Integer n = col_index.size();
  this->m_n.add(n);
  this->m_count += n;
  for (Integer i = 0; i < n; ++i) {
    this->m_row_index.add(row_index[i]);
    this->m_col_index.add(col_index[i]);
  }
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntries(
ConstArrayView<Integer> row_indexes,
const UniqueArray<ConstArrayView<Integer>>& col_indexes)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  // ALIEN_ASSERT((row_indexes.size()==col_indexes.size()),("row and col with different
  // sizes")) ;
  Integer n = 0;
  for (Integer i = 0; i < row_indexes.size(); ++i)
    n += col_indexes[i].size();
  this->m_n.add(n);
  this->m_count += n;
  for (Integer i = 0; i < row_indexes.size(); ++i) {
    Integer row = row_indexes[i];
    for (Integer j = 0; j < col_indexes[i].size(); ++j) {
      this->m_row_index.add(row);
      this->m_col_index.add(col_indexes[i][j]);
    }
  }
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntries(
ConstArrayView<Integer> row_indexes, UniqueArray2<Integer> col_indexes,
ConstArrayView<Integer> stencil_lids, Integer size)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  // ALIEN_ASSERT((row_indexes.size()==size),("row and col with different sizes")) ;
  Integer n = 0;
  for (Integer i = 0; i < stencil_lids.size(); ++i)
    n += size;
  this->m_n.add(n);
  this->m_count += n;
  for (Integer j = 0; j < stencil_lids.size(); ++j) {
    ArrayView<Integer> cols = col_indexes[stencil_lids[j]];
    for (Integer i = 0; i < size; ++i) {
      this->m_row_index.add(row_indexes[i]);
      this->m_col_index.add(cols[i]);
    }
  }
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntries(
const Integer row_index, ConstArrayView<Integer> col_index)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  Integer n = col_index.size();
  this->m_n.add(n);
  this->m_count += n;
  for (Integer i = 0; i < n; ++i) {
    this->m_row_index.add(row_index);
    this->m_col_index.add(col_index[i]);
  }
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntry(
Integer row_index, Integer col_index)
{
  this->_startTimer();
  ALIEN_ASSERT((this->m_parent->m_state == ePrepared), ("Inconsistent state"));
  this->m_n.add(1);
  ++this->m_count;
  this->m_row_index.add(row_index);
  this->m_col_index.add(col_index);
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::setData(ConstArrayView<ValueT> values)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()==this->m_current_size),("Incompatible size")) ;
  for (Integer i = 0; i < this->m_current_size; ++i)
    this->m_values[this->m_current_k[i]] = values[i];

  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::setData(ValueT values)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((1==this->m_current_size),("Incompatible size")) ;
  this->m_values[*this->m_current_k] = values;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addData(ConstArrayView<ValueT> values)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()==this->m_current_size),("Incompatible size")) ;
  for (Integer i = 0; i < this->m_current_size; ++i)
    this->m_values[this->m_current_k[i]] += values[i];
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addData(
ConstArrayView<ValueT> values, ValueT factor)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()==this->m_current_size),("Incompatible size")) ;
  for (Integer i = 0; i < this->m_current_size; ++i)
    this->m_values[this->m_current_k[i]] += values[i] * factor;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addData(ValueT values)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((1==this->m_current_size),("Incompatible size")) ;
  this->m_values[*this->m_current_k] += values;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addMultiData(ValueT value)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  for (Integer i = 0; i < this->m_current_size; ++i)
    this->m_values[this->m_current_k[i]] += value;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addBlockData(ConstArrayView<ValueT> values)
{
  this->_startTimer();
  ALIEN_ASSERT((this->m_parent->m_state == eStart), ("Inconsistent state"));
  ALIEN_ASSERT((values.size() == this->m_current_size * this->m_block_size),
               ("Incompatible size %d vs %d * %d ", values.size(), this->m_current_size,
                this->m_block_size));
  for (Integer i = 0; i < this->m_current_size; ++i)
    for (Integer k = 0; k < this->m_block_size; ++k)
      this->m_values[this->m_current_k[i] * this->m_block_size + k] +=
      values[i * this->m_block_size + k];
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addBlockData(
ConstArrayView<ValueT> values, ValueT factor)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()==this->m_current_size*this->m_block_size),("Incompatible
  // size")) ;
  for (Integer i = 0; i < this->m_current_size; ++i)
    for (Integer k = 0; k < this->m_block_size; ++k)
      this->m_values[this->m_current_k[i] * this->m_block_size + k] +=
      factor * values[i * this->m_block_size + k];
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addMultiData(
ConstArrayView<ValueT> values, Integer size)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()*size==this->m_current_size),("Incompatible size")) ;

  Integer icount = 0;
  for (Integer i = 0; i < values.size(); ++i)
    for (Integer k = 0; k < size; ++k)
      this->m_values[this->m_current_k[icount++]] += values[i];
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::Filler::addMultiData(
ConstArrayView<ValueT> values, ValueT factor, Integer size)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.size()*size==this->m_current_size),("Incompatible size")) ;
  Integer icount = 0;
  for (Integer i = 0; i < values.size(); ++i)
    for (Integer k = 0; k < size; ++k)
      this->m_values[this->m_current_k[icount++]] += factor * values[i];
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamMatrixBuilderT<ValueT>::Filler&
StreamMatrixBuilderT<ValueT>::Filler::operator++()
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  ++this->m_index;
  this->m_current_k += this->m_current_size;
  this->m_current_size = this->m_n[this->m_index];
  this->_stopTimer();
  return *this;
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
