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

/*!
 * \file MultiMatrixImpl.cc
 * \brief MultiMatrixImpl.cc
 */

#include "MultiMatrixImpl.h"

#include <alien/core/block/Block.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <cstdlib>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl::MultiMatrixImpl()
: m_block(nullptr)
, m_rows_block(nullptr)
, m_cols_block(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl::MultiMatrixImpl(std::shared_ptr<ISpace> row_space,
                                 std::shared_ptr<ISpace> col_space, std::shared_ptr<MatrixDistribution> dist)
: m_row_space(row_space)
, m_col_space(col_space)
, m_distribution(dist)
, m_block(nullptr)
, m_rows_block(nullptr)
, m_cols_block(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl::MultiMatrixImpl(const MultiMatrixImpl& impl)
: TimestampMng(impl)
, m_row_space(impl.m_row_space)
, m_col_space(impl.m_col_space)
, m_distribution(impl.m_distribution)
, m_block(nullptr)
, m_rows_block(nullptr)
, m_cols_block(nullptr)
{
  if (impl.m_block)
    m_block = impl.m_block;
  if (impl.m_rows_block)
    m_rows_block = impl.m_rows_block;
  if (impl.m_cols_block)
    m_cols_block = impl.m_cols_block;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl::~MultiMatrixImpl()
{
  free();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::setBlockInfos(Integer block_size)
{
  m_block.reset(new Block(block_size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::setBlockInfos(const Block* blocks)
{
  m_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::setBlockInfos(const VBlock* blocks)
{
  m_rows_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::setRowBlockInfos(VBlock const* blocks)
{
  m_rows_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::setColBlockInfos(VBlock const* blocks)
{
  m_cols_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::free()
{
  for (auto i = m_impls2.begin(); i != m_impls2.end(); ++i) {
    delete i->second;
    i->second = NULL;
  }
  m_impls2.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::clear()
{
  for (auto i = m_impls2.begin(); i != m_impls2.end(); ++i) {
    i->second->clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Block*
MultiMatrixImpl::block() const
{
  return m_block.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
MultiMatrixImpl::rowBlock() const
{
  return m_rows_block.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
MultiMatrixImpl::colBlock() const
{
  if (m_cols_block.get())
    return m_cols_block.get();
  else
    return rowBlock();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
MultiMatrixImpl::vblock() const
{
  return m_rows_block.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMatrixImpl*
MultiMatrixImpl::clone() const
{
  // Initialize muli-representation without data
  auto impl = new MultiMatrixImpl(*this);

  typedef BackEnd::tag::simplecsr tag;
  // We get the last up to date implementation
  const SimpleCSRMatrix<Real>& matrixToClone = this->get<tag>();
  // And clone it
  SimpleCSRMatrix<Real>* matrixCloned = matrixToClone.cloneTo(impl);
  matrixCloned->setTimestamp(impl, matrixToClone.timestamp());
  matrixCloned->updateTimestamp();
  impl->m_impls2.insert(
  MultiMatrixImplMap::value_type(AlgebraTraits<tag>::name(), matrixCloned));

  // TOCHECK: to be removed or not ?
  /* WARNING: this implementation is temporary. Later it should be implemented through a
     clone() method for each kernel as written lower.
     For now, we will convert the last up to date matrix in SimpleCSRMatrix, then clone
     it.

     for(std::map<BackEndId, IMatrixImpl*>::const_iterator it = m_impls2.begin(); it !=
     m_impls2.end(); ++it)
     impl->m_impls2.insert(MultiMatrixImplMap::value_type(it->first,
     it->second->clone()));
  */
  return impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMatrixImpl::updateImpl(IMatrixImpl* target) const
{
  // If we are up to date, do nothing
  if (timestamp() == target->timestamp()) {
    return;
  }

  // Otherwise we are looking for a converter
  for (auto& impl : m_impls2) {
    auto* candidate = impl.second;
    // if the timestamp is not good we keep looking
    if (candidate->timestamp() != timestamp())
      continue;
    auto* converter =
    MatrixConverterRegisterer::getConverter(candidate->backend(), target->backend());
    // If no converter is found, we continue
    if (converter == nullptr)
      continue;
    // Otherwise we convert the matrix and return
    converter->convert(candidate, target);
    target->copyTimestamp(*candidate);
    return;
  }

  // If there is no candidate up to date, throw an error
  UniqueArray<IMatrixImpl*> candidates;
  for (auto& impl : m_impls2) {
    auto* candidate = impl.second;
    if (candidate->timestamp() == timestamp())
      candidates.add(candidate);
  }
  if (candidates.empty()) {
    auto msg = String::format("Matrix converter to: ", target->backend(),
                              " internal error: no timestamp matching");
    throw FatalErrorException(A_FUNCINFO, msg);
  }

  // Otherwise, we have candidates but no converter
  // In that case we try to use simplecsr as a third party converter

  // Error message that will be printed
  auto print_error = [&] {
    alien_fatal([&] {
      cout() << "ALIEN FATAL ERROR \n"
             << "Matrix converting to target backend '" << target->backend() << "' :\n"
             << "* no converter available from source backend(s)";

      for (auto* candidate : candidates) {
        cout() << "   ** '" << candidate->backend() << "'";
      }
      cout();
    });
  };

  // Request simplecsr implementation
  auto* simplecsr = getImpl<SimpleCSRMatrix<Real>>("simplecsr");

  // Checking that we have a converter from simplecsr to the requested implementation
  auto* simplecsr_target =
  MatrixConverterRegisterer::getConverter(simplecsr->backend(), target->backend());

  // If not, throw error
  if (simplecsr_target == nullptr)
    print_error();

  // We look for each candidate
  for (auto* candidate : candidates) {
    auto* candidat_simplecsr = MatrixConverterRegisterer::getConverter(
    candidate->backend(), simplecsr->backend());
    // If we have one, we can convert to the requested implementation through simplecsr
    if (candidat_simplecsr != nullptr) {
      // Conversion from candidate to simplecsr
      candidat_simplecsr->convert(candidate, simplecsr);
      simplecsr->copyTimestamp(*candidate);
      // Conversion from simplecsr to target
      simplecsr_target->convert(simplecsr, target);
      target->copyTimestamp(*simplecsr);
      return;
    }
  }

  // If we reach this line, all conversions possibilities have failed. Throw an exception
  print_error();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
