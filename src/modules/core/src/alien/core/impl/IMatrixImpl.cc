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
 * \file IMatrixImpl.cc
 * \brief IMatrixImpl.cc
 */

#include "IMatrixImpl.h"

#include <alien/core/block/Block.h>
#include <alien/core/impl/MultiMatrixImpl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMatrixImpl::IMatrixImpl(const MultiMatrixImpl* multi_impl, BackEndId backend)
: Timestamp(multi_impl)
, m_multi_impl(multi_impl)
, m_backend(backend)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
IMatrixImpl::rowSpace() const
{
  return m_multi_impl->rowSpace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
IMatrixImpl::colSpace() const
{
  return m_multi_impl->colSpace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const MatrixDistribution&
IMatrixImpl::distribution() const
{
  return m_multi_impl->distribution();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Block*
IMatrixImpl::block() const
{
  return m_multi_impl ? m_multi_impl->block() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
IMatrixImpl::vblock() const
{
  return m_multi_impl ? m_multi_impl->vblock() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
IMatrixImpl::rowBlock() const
{
  return m_multi_impl ? m_multi_impl->rowBlock() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock*
IMatrixImpl::colBlock() const
{
  return m_multi_impl ? m_multi_impl->colBlock() : nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
