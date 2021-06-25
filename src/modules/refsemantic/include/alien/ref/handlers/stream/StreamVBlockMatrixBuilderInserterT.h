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

/*---------------------------------------------------------------------------*/

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::BaseInserter()
: m_parent(NULL)
{
  throw FatalErrorException(A_FUNCINFO, "Virtual Ctor never called");
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::BaseInserter(
StreamVBlockMatrixBuilderT<ValueT>* parent, Integer id)
: m_id(id)
, m_index(0)
, m_current_size(0)
, m_current_k(NULL)
, m_values(NULL)
, m_count(0)
, m_size(0)
, m_parent(parent)
{}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::~BaseInserter()
{}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::init()
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
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::getId() const
{
  return m_id;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::end()
{
  init();
  m_n.resize(0);
  m_row_index.resize(0);
  m_col_index.resize(0);
  m_data_index.resize(0);
  m_block_size_row.resize(0);
  m_block_size_col.resize(0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::setMatrixValues(ValueT* matrix_values)
{
  ALIEN_ASSERT((this->m_parent->m_state == ePrepared), ("Inconsistent state"));
  m_values = matrix_values;
  m_index = 0;
  m_current_size = m_n[0];
  m_current_k = m_data_index.data();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::size()
{
  return m_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamVBlockMatrixBuilderT<ValueT>::BaseInserter::count()
{
  return m_count;
}

template <typename ValueT>
bool StreamVBlockMatrixBuilderT<ValueT>::Filler::isBegin()
{
  return (this->m_index == 0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
bool StreamVBlockMatrixBuilderT<ValueT>::Filler::isEnd()
{
  return (this->m_index == this->m_size);
}

template <typename ValueT>
Integer
StreamVBlockMatrixBuilderT<ValueT>::Filler::currentSize()
{
  return this->m_current_size;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
Integer
StreamVBlockMatrixBuilderT<ValueT>::Filler::index()
{
  return this->m_index;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::Filler::start()
{
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((this->m_values!=NULL),("Inserter is not ready for filling"));
  this->m_index = 0;
  this->m_current_size = this->m_n[0];
  this->m_current_k = this->m_data_index.data();
  this->m_current_block_size_row = this->m_block_size_row[0];
  this->m_current_block_size_col = this->m_block_size_col[0];
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::Profiler::addMatrixEntry(
Integer row_index, Integer col_index)
{
  // std::cout << "inserter inside " << row_index << "," << col_index << std::endl;
  this->_startTimer();
  const VBlock* block_sizes = this->m_parent->vblock();
  // ALIEN_ASSERT((this->m_parent->m_state == ePrepared),("Inconsistent state"));
  this->m_n.add(1);
  ++this->m_count;
  this->m_row_index.add(row_index);
  this->m_col_index.add(col_index);
  this->m_block_size_row.add(block_sizes->size(row_index));
  this->m_block_size_col.add(block_sizes->size(col_index));
  ++this->m_size;
  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::Filler::addBlockData(ConstArray2View<ValueT> values)
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  // ALIEN_ASSERT((values.dim1Size()==this->m_current_block_size_row),("Incompatible block
  // row size %d vs %d",values.dim1Size(),this->m_current_block_size_row));
  // ALIEN_ASSERT((values.dim2Size()==this->m_current_block_size_col),("Incompatible block
  // row size %d vs %d",values.dim2Size(),this->m_current_block_size_col));

  Array2View<ValueT> view(
  this->m_values + *this->m_current_k, values.dim1Size(), values.dim2Size());

  for (Integer i = 0; i < values.dim1Size(); ++i)
    for (Integer k = 0; k < values.dim2Size(); ++k)
      view[i][k] += values[i][k];

  this->_stopTimer();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamVBlockMatrixBuilderT<ValueT>::Filler&
StreamVBlockMatrixBuilderT<ValueT>::Filler::operator++()
{
  this->_startTimer();
  // ALIEN_ASSERT((this->m_parent->m_state == eStart),("Inconsistent state"));
  ++this->m_index;
  this->m_current_k += this->m_current_size;
  this->m_current_size = this->m_n[this->m_index];
  this->m_current_block_size_row = this->m_block_size_row[this->m_index];
  this->m_current_block_size_col = this->m_block_size_col[this->m_index];
  this->_stopTimer();
  return *this;
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
