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
 * \file MatrixDistribution.h
 * \brief MatrixDistribution.h
 */

#pragma once

#include <arccore/base/BaseTypes.h>

#include <alien/data/ISpace.h>

namespace Arccore::MessagePassing
{
class IMessagePassingMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorDistribution;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup distribution
 * \brief Computes a matrix distribution
 *
 * Computes or use a pre-existing block row distribution for matrices
 */
class ALIEN_EXPORT MatrixDistribution
{
 public:
  //! Constructor
  MatrixDistribution();

  /*!
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(const ISpace& row_space, const ISpace& col_space,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(const ISpace& row_space, const ISpace& col_space,
                     Integer local_row_size, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(const ISpace& row_space, const ISpace& col_space,
                     Integer local_row_size, Integer local_col_size, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     Arccore::Integer local_row_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     Arccore::Integer local_row_size, Arccore::Integer local_col_size,
                     Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     Arccore::Integer local_row_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  MatrixDistribution(Arccore::Integer global_row_size, Arccore::Integer global_col_size,
                     Arccore::Integer local_row_size, Arccore::Integer local_col_size,
                     std::shared_ptr<Arccore::MessagePassing::IMessagePassingMng> parallel_mng);

  /*!
   * \brief Copy constructor
   * \param[in] dist The matrix distribution to copy
   */
  MatrixDistribution(const MatrixDistribution& dist);

  /*!
   * \brief Rvalue constructor
   * \param[in] dist The distribution to take
   */
  MatrixDistribution(MatrixDistribution&& dist);

  //! Free resources
  ~MatrixDistribution();

  /*!
   * \brief Operator equal
   * \param[in] dist The matrix distribution to copy
   * \returns The new matrix distribution
   */
  MatrixDistribution& operator=(const MatrixDistribution& dist);

  /*!
   * \brief Operator equal
   * \param[in] dist The matrix distribution to copy
   * \returns The new matrix distribution
   */
  MatrixDistribution& operator=(MatrixDistribution&& dist);

  /*!
   * \brief Comparison operator
   * \param[in] dist The matrix distribution to compare
   * \returns Whether or not the matrices distribution are the same
   */
  bool operator==(const MatrixDistribution& dist) const;

  /*!
   * \brief Whether or not the run is parallel
   * \returns Whether or not the run is parallel
   */
  bool isParallel() const;

  /*!
   * \brief Get the parallel manager
   * \returns The parallel manager
   */
  Arccore::MessagePassing::IMessagePassingMng* parallelMng() const;

  /*!
   * \brief Get the row distribution
   * \returns The row distribution
   */
  const VectorDistribution& rowDistribution() const;

  /*!
   * \brief Get the col distribution
   * \returns The col distribution
   */
  const VectorDistribution& colDistribution() const;

  /*!
   * \brief Get the row space
   * \returns The row space
   */
  const ISpace& rowSpace() const;

  /*!
   * \brief Get the col space
   * \returns The col space
   */
  const ISpace& colSpace() const;

  /*!
   * \brief Get the local row size
   * \returns The local row size
   */
  Arccore::Integer localRowSize() const;

  /*!
   * \brief Get the local col size
   * \returns The local col size
   */
  Arccore::Integer localColSize() const;

  /*!
   * \brief Get the global row size
   * \returns The global row size
   */
  Arccore::Integer globalRowSize() const;

  /*!
   * \brief Get the global col size
   * \returns The global col size
   */
  Arccore::Integer globalColSize() const;

  /*!
   * \brief Get the row offset
   * \return The row offset
   */
  Arccore::Integer rowOffset() const;

  /*!
   * \brief Get the row offset for a specific proc
   * \param[in] p The requested proc
   * \returns The row offset for the specific proc
   */
  Arccore::Integer rowOffset(Arccore::Integer p) const;

  /*!
   * \brief Get the col offset
   * \return The col offset
   */
  Arccore::Integer colOffset() const;

  /*!
   * \brief Get the col offset for a specific proc
   * \param[in] p The requested proc
   * \returns The col offset for the specific proc
   */
  Arccore::Integer colOffset(Arccore::Integer p) const;

  /*!
   * \brief Clone the distribution
   * \returns A clone of this distribution
   */
  std::shared_ptr<MatrixDistribution> clone() const;

 private:
  /*!
   * \brief Get the local id of a row with its global id
   * \param[in] i The global id of the row
   * \returns The local id of the row
   */
  Arccore::Integer rowGlobalToLocal(Arccore::Integer i) const;

  /*!
   * \brief Get the local id of a non local row with its global id
   * \param[in] i The global id of the non local row
   * \param[in] p The proc which owns the row
   * \returns The local id of the row
   */
  Arccore::Integer rowGlobalToLocal(Arccore::Integer i, Arccore::Integer p) const;

  /*!
   * \brief Get the local id of a col with its global id
   * \param[in] i The global id of the col
   * \returns The local id of the col
   */
  Arccore::Integer colGlobalToLocal(Arccore::Integer i) const;

  /*!
   * \brief Get the local id of a non local col with its global id
   * \param[in] i The global id of the non local col
   * \param[in] p The proc which owns the col
   * \returns The local id of the col
   */
  Arccore::Integer colGlobalToLocal(Arccore::Integer i, Arccore::Integer p) const;

  /*!
   * \brief Get the global id of a row with its local id
   * \param[in] i The local id of the row
   * \returns The global id of the row
   */
  Arccore::Integer rowLocalToGlobal(Arccore::Integer i) const;

  /*!
   * \brief Get the global id of a non local row with its local id
   * \param[in] i The local id of the non local row
   * \param[in] p The proc which owns the row
   * \returns The global id of the row
   */
  Arccore::Integer rowLocalToGlobal(Arccore::Integer i, Arccore::Integer p) const;

  /*!
   * \brief Get the global id of a col with its local id
   * \param[in] i The local id of the col
   * \returns The global id of the col
   */
  Arccore::Integer colLocalToGlobal(Arccore::Integer i) const;

  /*!
   * \brief Get the global id of a non local col with its local id
   * \param[in] i The local id of the non local col
   * \param[in] p The proc which owns the col
   * \returns The global id of the col
   */
  Arccore::Integer colLocalToGlobal(Arccore::Integer i, Arccore::Integer p) const;

  /*!
   * \brief Get the owner of an entry
   * \param[in] i The global id of the row
   * \param[in] j The global id of the col
   * \returns The proc owner
   */
  Arccore::Integer owner(Arccore::Integer i, Arccore::Integer j) const;

 private:
  struct Internal;
  //! Internal implementation of the matrix distribution
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Print the distribution
extern ALIEN_EXPORT std::ostream& operator<<(
std::ostream& nout, const MatrixDistribution& dist);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
