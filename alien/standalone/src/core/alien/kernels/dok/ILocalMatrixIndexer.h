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

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <optional>

#include <alien/utils/Precomp.h>

namespace Alien
{

class IReverseIndexer;

//! Interface for a local sparse matrix indexer.
//! The goal of this indexer is to provide a database indexed by (row,column) ids.
//! Non-zeros are not stored here, only their position in the nnz storage array.
//! This interface allows to use a CSRMatrix as a DoKMatrix.
class ILocalMatrixIndexer
{
 public:
  typedef Integer Offset;
  typedef std::pair<Offset, Offset> Renumbering;
  typedef std::pair<Int32, Int32> Key;

 public:
  virtual ~ILocalMatrixIndexer() {}

  //! Registers an offset with a matrix position (i,j)
  //! \param i id of the row
  //! \param j id of the column
  //! \param offset
  virtual void associate(Integer i, Integer j, Offset offset) = 0;

  //! Finds the offset associated with a matrix position (i,j)
  //! \param i
  //! \param j
  //! \return offset if found
  virtual std::optional<Offset> find(Integer i, Integer j) = 0;

  //! Creates a new offset for matrix position (i,j)
  //! \param i
  //! \param j
  //! \param tentative_offset hint on what the offset should be
  //! \return the offset (can be different from tentative_offset)
  virtual Offset create(Integer i, Integer j, Offset& tentative_offset) = 0;

  //! Creates a new indexer, based on offset permutations.
  //! \param perm permutation array, to be filled by this function.
  //! \return new indexer, used to compact matrix
  virtual IReverseIndexer* sort(Arccore::Array<Renumbering>& perm) = 0;

  virtual ILocalMatrixIndexer* clone() const = 0;
};

} // namespace Alien
