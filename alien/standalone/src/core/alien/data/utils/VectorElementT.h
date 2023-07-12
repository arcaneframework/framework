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
 * \file VectorElementT.h
 * \brief VectorElementT.h
 */

#pragma once

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
VectorElementT<T, Indexer>::VectorElementT(Arccore::ArrayView<T> values,
                                           Arccore::ConstArrayView<Arccore::Integer> indexes,
                                           const Arccore::Integer local_offset)
: m_values(values)
, m_indexes(indexes)
, m_local_offset(local_offset)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void VectorElementT<T, Indexer>::operator=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] = values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void VectorElementT<T, Indexer>::operator+=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] += values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void VectorElementT<T, Indexer>::operator-=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] -= values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
MultVectorElementT<T, Indexer>::MultVectorElementT(Arccore::ArrayView<T> values,
                                                   T factor,
                                                   Arccore::ConstArrayView<Arccore::Integer> indexes,
                                                   const Arccore::Integer local_offset)
: m_values(values)
, m_factor(factor)
, m_indexes(indexes)
, m_local_offset(local_offset)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElementT<T, Indexer>::operator=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] = m_factor * values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElementT<T, Indexer>::operator+=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] += m_factor * values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElementT<T, Indexer>::operator-=(Arccore::ConstArrayView<T> values)
{
  ALIEN_ASSERT((m_indexes.size() == values.size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i], m_local_offset);
    m_values[id] -= m_factor * values[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
MultVectorElement2T<T, Indexer>::MultVectorElement2T(Arccore::ArrayView<T> values,
                                                     T factor,
                                                     Arccore::ConstArray2View<Arccore::Integer> indexes,
                                                     Arccore::Integer i,
                                                     const Arccore::Integer local_offset)
: m_values(values)
, m_factor(factor)
, m_indexes(indexes)
, m_i(i)
, m_local_offset(local_offset)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElement2T<T, Indexer>::operator=(Arccore::ConstArray2View<T> values)
{
  ALIEN_ASSERT((m_indexes.dim1Size() == values.dim1Size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.dim1Size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i][m_i], m_local_offset);
    m_values[id] = m_factor * values[i][m_i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElement2T<T, Indexer>::operator+=(Arccore::ConstArray2View<T> values)
{
  ALIEN_ASSERT((m_indexes.dim1Size() == values.dim1Size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.dim1Size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i][m_i], m_local_offset);
    m_values[id] += m_factor * values[i][m_i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename Indexer>
void MultVectorElement2T<T, Indexer>::operator-=(Arccore::ConstArray2View<T> values)
{
  ALIEN_ASSERT((m_indexes.dim1Size() == values.dim1Size()), ("Invalid size"));
  for (Arccore::Integer i = 0; i < m_indexes.dim1Size(); ++i) {
    const Arccore::Integer id = Indexer::index(m_indexes[i][m_i], m_local_offset);
    m_values[id] -= m_factor * values[i][m_i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
