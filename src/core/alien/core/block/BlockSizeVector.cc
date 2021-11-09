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
 * \file BlockSizeVector.cc
 * \brief BlockSizeVector.cc
 */

#include "BlockSizeVector.h"

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizeVector::BlockSizeVector(
UniqueArray<Integer>& sizes, Integer offset, ConstArrayView<Integer> indexes)
: m_sizes(sizes)
, m_offset(offset)
, m_indexes(indexes)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizeVector&
BlockSizeVector::operator=(Integer size)
{
  for (Integer i = 0; i < m_indexes.size(); ++i) {
    m_sizes[m_indexes[i] - m_offset] = size;
  }
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizeVector&
BlockSizeVector::operator+=(Integer size)
{
  for (Integer i = 0; i < m_indexes.size(); ++i) {
    m_sizes[m_indexes[i] - m_offset] += size;
  }
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizeVector
BlockSizeVectorFiller::operator[](ConstArrayView<Integer> indexes)
{
  return BlockSizeVector(m_sizes, m_offset, indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BlockSizeVector
BlockSizeVectorFiller::operator[](ConstArray2View<Integer> indexes)
{
  ConstArrayView<Integer> ids(indexes.totalNbElement(), indexes.unguardedBasePointer());
  return BlockSizeVector(m_sizes, m_offset, ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer&
BlockSizeVectorFiller::operator[](Integer index)
{
  return m_sizes[index - m_offset];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
