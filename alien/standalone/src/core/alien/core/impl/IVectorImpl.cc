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
 * \file IVectorImpl.cc
 * \brief IVectorImpl.cc
 */

#include "IVectorImpl.h"

#include <alien/core/block/Block.h>
#include <alien/core/impl/MultiVectorImpl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVectorImpl::IVectorImpl(const MultiVectorImpl* multi_impl, BackEndId backend)
: Timestamp(multi_impl)
, m_multi_impl(multi_impl)
, m_backend(backend)
, m_vblock_sizes(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
IVectorImpl::space() const
{
  return m_multi_impl->space();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VectorDistribution&
IVectorImpl::distribution() const
{
  return m_multi_impl->distribution();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Block*
IVectorImpl::block() const
{
  return m_multi_impl ? m_multi_impl->block() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
IVectorImpl::vblock() const
{
  return m_multi_impl ? m_multi_impl->vblock() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
IVectorImpl::scalarizedLocalSize() const
{
  auto& dist = m_multi_impl->distribution();
  Integer local_size = dist.localSize();
  if (block())
    local_size *= block()->size();
  else if (vblock()) {
    if (m_vblock_sizes == nullptr)
      m_vblock_sizes = new VBlockSizes(*vblock(), dist);
    local_size = m_vblock_sizes->localSize();
  }
  return local_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
IVectorImpl::scalarizedGlobalSize() const
{
  auto& dist = m_multi_impl->distribution();
  Integer global_size = dist.globalSize();
  if (block())
    global_size *= block()->size();
  else if (vblock()) {
    if (m_vblock_sizes == nullptr)
      m_vblock_sizes = new VBlockSizes(*vblock(), dist);
    global_size = m_vblock_sizes->globalSize();
  }
  return global_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
IVectorImpl::scalarizedOffset() const
{
  auto& dist = m_multi_impl->distribution();
  Integer offset = dist.offset();
  if (block())
    offset *= block()->size();
  else if (vblock()) {
    if (m_vblock_sizes == nullptr)
      m_vblock_sizes = new VBlockSizes(*vblock(), dist);
    offset = m_vblock_sizes->offset();
  }
  return offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
