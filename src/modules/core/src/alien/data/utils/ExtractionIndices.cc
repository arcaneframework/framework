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
 * \file ExtractionIndices.cc
 * \brief ExtractionIndices.cc
 */

#include "ExtractionIndices.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExtractionIndices::ExtractionIndices(const Integer rowStart, const Integer rowRange,
                                     const Integer colStart, const Integer colRange)
: m_starting_row(rowStart)
, m_starting_col(colStart)
, m_row_range(rowRange)
, m_col_range(colRange)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExtractionIndices::~ExtractionIndices() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::rowStart() const
{
  return m_starting_row;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::rowRange() const
{
  return m_row_range;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::colStart() const
{
  return m_starting_col;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::colRange() const
{
  return m_col_range;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::toLocalRow(const Integer uid) const
{
  return uid - m_starting_row;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
ExtractionIndices::toLocalCol(const Integer uid) const
{
  return uid - m_starting_col;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
