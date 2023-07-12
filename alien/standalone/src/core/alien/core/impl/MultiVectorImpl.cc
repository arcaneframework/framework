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
 * \file MultiVectorImpl.cc
 * \brief MultiVectorImpl.cc
 */

#include "MultiVectorImpl.h"

#include <alien/kernels/simple_csr/SimpleCSRVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl::MultiVectorImpl()
: m_block(nullptr)
, m_variable_block(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl::MultiVectorImpl(
std::shared_ptr<ISpace> space, std::shared_ptr<VectorDistribution> distribution)
: m_space(std::move(space))
, m_distribution(distribution)
, m_block(nullptr)
, m_variable_block(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl::MultiVectorImpl(const MultiVectorImpl& impl)
: TimestampMng(impl)
, m_space(impl.m_space)
, m_distribution(impl.m_distribution)
, m_block(nullptr)
, m_variable_block(nullptr)
{
  if (impl.m_block)
    m_block = impl.m_block;
  if (impl.m_variable_block)
    m_variable_block = impl.m_variable_block;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl::~MultiVectorImpl()
{
  free();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiVectorImpl::setBlockInfos(const Block* blocks)
{
  m_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiVectorImpl::setBlockInfos(const VBlock* blocks)
{
  m_variable_block = blocks->clone();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiVectorImpl::setBlockInfos(Integer block_size)
{
  m_block.reset(new Block(block_size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiVectorImpl::free()
{
  for (MultiVectorImplMap::iterator i = m_impls2.begin(); i != m_impls2.end(); ++i) {
    delete i->second;
    i->second = NULL;
  }
  m_impls2.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiVectorImpl::clear()
{
  for (MultiVectorImplMap::iterator i = m_impls2.begin(); i != m_impls2.end(); ++i) {
    i->second->clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Block*
MultiVectorImpl::block() const
{
  return m_block.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
MultiVectorImpl::vblock() const
{
  return m_variable_block.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiVectorImpl*
MultiVectorImpl::clone() const
{
  // Initialize muli-representation without data
  auto impl = new MultiVectorImpl(*this);

  // We get the last up to date implementation
  typedef BackEnd::tag::simplecsr tag;
  const SimpleCSRVector<Real>& vectorToClone = this->get<tag>();
  // And clone it
  SimpleCSRVector<Real>* vectorCloned = vectorToClone.cloneTo(impl);
  vectorCloned->setTimestamp(impl, vectorToClone.timestamp());
  vectorCloned->updateTimestamp();
  impl->m_impls2.insert(
  MultiVectorImplMap::value_type(AlgebraTraits<tag>::name(), vectorCloned));

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

void MultiVectorImpl::updateImpl(IVectorImpl* target) const
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
    VectorConverterRegisterer::getConverter(candidate->backend(), target->backend());
    // If no converter is found, we continue
    if (converter == nullptr)
      continue;
    // Otherwise we convert the vector and return
    converter->convert(candidate, target);
    target->copyTimestamp(*candidate);
    return;
  }

  // If there is no candidate up to date, throw an error
  UniqueArray<IVectorImpl*> candidates;
  for (auto& impl : m_impls2) {
    auto* candidate = impl.second;
    if (candidate->timestamp() == timestamp())
      candidates.add(candidate);
  }
  if (candidates.empty()) {
    auto msg = String::format("Vector converter to: ", target->backend(),
                              " internal error: no timestamp matching");
    throw FatalErrorException(A_FUNCINFO, msg);
  }

  // Otherwise, we have candidates but no converter
  // In that case we try to use simplecsr as a third party converter

  // Error message that will be printed
  auto print_error = [&] {
    alien_fatal([&] {
      cout() << "ALIEN FATAL ERROR \n"
             << "Vector converting to target backend '" << target->backend() << "' :\n"
             << "* no converter available from source backend(s)";

      for (auto* candidate : candidates) {
        cout() << "   ** '" << candidate->backend() << "'";
      }
      cout();
    });
  };

  // Request simplecsr implementation
  auto* simplecsr = getImpl<SimpleCSRVector<Real>>("simplecsr");

  // Checking that we have a converter from simplecsr to the requested implementation
  auto* simplecsr_target =
  VectorConverterRegisterer::getConverter(simplecsr->backend(), target->backend());

  // If not, throw error
  if (simplecsr_target == nullptr)
    print_error();

  // We look for each candidate
  for (auto* candidate : candidates) {
    auto* candidat_simplecsr = VectorConverterRegisterer::getConverter(
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
