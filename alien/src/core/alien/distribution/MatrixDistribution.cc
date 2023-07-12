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
 * \file MatrixDistribution.cc
 * \brief MatrixDistribution.cc
 */

#include "MatrixDistribution.h"

#include <iostream>

#include <alien/data/Space.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/utils/Trace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal implementation of the matrix distribution
struct MatrixDistribution::Internal
{
  //! Constructor
  Internal();

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(
  const ISpace& row_space, const ISpace& col_space, IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& row_space, const ISpace& col_space,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(
  Integer global_row_size, Integer global_col_size, IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_row_size, Integer global_col_size,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& row_space, const ISpace& col_space, Integer local_row_size,
           IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& row_space, const ISpace& col_space, Integer local_row_size,
           Integer local_col_size, IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& row_space, const ISpace& col_space, Integer local_row_size,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*
   * \brief Constructor
   * \param[in] row_space The row space of the matrix
   * \param[in] col_space The col space of the matrix
   * \param[in] local_row_size The number of local rows in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& row_space, const ISpace& col_space, Integer local_row_size,
           Integer local_col_size, std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_row_size, Integer global_col_size, Integer local_row_size,
           IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_row_size, Integer global_col_size, Integer local_row_size,
           Integer local_col_size, IMessagePassingMng* parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_row_size, Integer global_col_size, Integer local_row_size,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*
   * \brief Constructor
   * \param[in] global_row_size The number of rows in the matrix
   * \param[in] global_col_size The number of cols in the matrix
   * \param[in] local_row_size The number of local cols in the matrix
   * \param[in] local_col_size The number of local cols in the matrix
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_row_size, Integer global_col_size, Integer local_row_size,
           Integer local_col_size, std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*!
   * \brief Comparison operator
   * \param[in] dist The matrix distribution to compare
   * \returns Whether or not the matrices distribution are the same
   */
  bool operator==(const Internal& dist) const;

  //! Row distribution
  std::shared_ptr<VectorDistribution> m_row_distribution;
  //! Col distribution
  std::shared_ptr<VectorDistribution> m_col_distribution;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal()
: m_row_distribution(std::make_shared<VectorDistribution>(VectorDistribution()))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(
const ISpace& row_space, const ISpace& col_space, IMessagePassingMng* parallel_mng)
: m_row_distribution(std::make_shared<VectorDistribution>(row_space, parallel_mng))
, m_col_distribution(std::make_shared<VectorDistribution>(col_space, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(const ISpace& row_space, const ISpace& col_space,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_row_distribution(std::make_shared<VectorDistribution>(row_space, parallel_mng))
, m_col_distribution(std::make_shared<VectorDistribution>(col_space, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(
Integer global_row_size, Integer global_col_size, IMessagePassingMng* parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(Integer global_row_size, Integer global_col_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, IMessagePassingMng* parallel_mng)
: m_row_distribution(
  std::make_shared<VectorDistribution>(row_space, local_row_size, parallel_mng))
, m_col_distribution(std::make_shared<VectorDistribution>(col_space, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, Integer local_col_size, IMessagePassingMng* parallel_mng)
: m_row_distribution(
  std::make_shared<VectorDistribution>(row_space, local_row_size, parallel_mng))
, m_col_distribution(
  std::make_shared<VectorDistribution>(col_space, local_col_size, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_row_distribution(
  std::make_shared<VectorDistribution>(row_space, local_row_size, parallel_mng))
, m_col_distribution(std::make_shared<VectorDistribution>(col_space, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, Integer local_col_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_row_distribution(
  std::make_shared<VectorDistribution>(row_space, local_row_size, parallel_mng))
, m_col_distribution(
  std::make_shared<VectorDistribution>(col_space, local_col_size, parallel_mng))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, IMessagePassingMng* parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), local_row_size, parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, Integer local_col_size, IMessagePassingMng* parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), local_row_size, local_col_size,
           parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, std::shared_ptr<IMessagePassingMng> parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), local_row_size, parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::Internal::Internal(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, Integer local_col_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: Internal(Space(global_row_size), Space(global_col_size), local_row_size, local_col_size,
           parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MatrixDistribution::Internal::operator==(const Internal& dist) const
{
  return ((dist.m_row_distribution == m_row_distribution) && (dist.m_col_distribution == m_col_distribution));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution()
: m_internal(std::make_shared<Internal>())
{
  alien_debug([&] { cout() << "Create Empty MatrixDistribution"; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(
const ISpace& row_space, const ISpace& col_space, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(row_space, col_space, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(rowglobal=" << row_space.size()
           << ",colglobal=" << col_space.size() << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(
Integer global_row_size, Integer global_col_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(global_row_size, global_col_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(rowglobal=" << global_row_size
           << ",colglobal=" << global_col_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(Integer global_row_size, Integer global_col_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(global_row_size, global_col_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(rowglobal=" << global_row_size
           << ",colglobal=" << global_col_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, IMessagePassingMng* parallel_mng)
: m_internal(
  std::make_shared<Internal>(row_space, col_space, local_row_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << row_space.size()
           << ",global_col=" << col_space.size() << ",local_row=" << local_row_size
           << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(const ISpace& row_space, const ISpace& col_space,
                                       Integer local_row_size, Integer local_col_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(
  row_space, col_space, local_row_size, local_col_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << row_space.size()
           << ",global_col=" << col_space.size() << ",local_row=" << local_row_size
           << ",local_col=" << local_col_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(
  global_row_size, global_col_size, local_row_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << global_row_size
           << ",global_col=" << global_col_size << ",local_row=" << local_row_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, Integer local_col_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(
  global_row_size, global_col_size, local_row_size, local_col_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << global_row_size
           << ",global_col=" << global_col_size << ",local_row=" << local_row_size
           << ",local_col=" << local_col_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(
  global_row_size, global_col_size, local_row_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << global_row_size
           << ",global_col=" << global_col_size << ",local_row=" << local_row_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(Integer global_row_size, Integer global_col_size,
                                       Integer local_row_size, Integer local_col_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(
  global_row_size, global_col_size, local_row_size, local_col_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create MatrixDistribution(global_row=" << global_row_size
           << ",global_col=" << global_col_size << ",local_row=" << local_row_size
           << ",local_col=" << local_col_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(const MatrixDistribution& dist) = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::MatrixDistribution(MatrixDistribution&& dist) = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution::~MatrixDistribution() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VectorDistribution&
MatrixDistribution::rowDistribution() const
{
  return *(m_internal->m_row_distribution);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
MatrixDistribution::rowSpace() const
{
  return m_internal->m_row_distribution->space();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VectorDistribution&
MatrixDistribution::colDistribution() const
{
  if (m_internal->m_col_distribution.get())
    return *(m_internal->m_col_distribution);
  return *(m_internal->m_row_distribution);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
MatrixDistribution::colSpace() const
{
  return colDistribution().space();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution&
MatrixDistribution::operator=(const MatrixDistribution& dist) = default;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixDistribution&
MatrixDistribution::operator=(MatrixDistribution&& dist) = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MatrixDistribution::operator==(const MatrixDistribution& dist) const
{
  return *m_internal == *dist.m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MatrixDistribution::isParallel() const
{
  return m_internal->m_row_distribution->isParallel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng*
MatrixDistribution::parallelMng() const
{
  return m_internal->m_row_distribution->parallelMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::localRowSize() const
{
  return m_internal->m_row_distribution->localSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::localColSize() const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->localSize();
  else
    return m_internal->m_row_distribution->localSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::globalRowSize() const
{
  return m_internal->m_row_distribution->globalSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::globalColSize() const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->globalSize();
  else
    return m_internal->m_row_distribution->globalSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowOffset() const
{
  return m_internal->m_row_distribution->offset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowOffset(Integer p) const
{
  return m_internal->m_row_distribution->offset(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colOffset() const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->offset();
  else
    return m_internal->m_row_distribution->offset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colOffset(Integer p) const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->offset(p);
  else
    return m_internal->m_row_distribution->offset(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<MatrixDistribution>
MatrixDistribution::clone() const
{
  return std::make_shared<MatrixDistribution>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowGlobalToLocal(Integer i) const
{
  return m_internal->m_row_distribution->globalToLocal(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowGlobalToLocal(Integer i, Integer p) const
{
  return m_internal->m_row_distribution->globalToLocal(i, p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowLocalToGlobal(Integer i) const
{
  return m_internal->m_row_distribution->localToGlobal(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::rowLocalToGlobal(Integer i, Integer p) const
{
  return m_internal->m_row_distribution->localToGlobal(i, p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colGlobalToLocal(Integer i) const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->globalToLocal(i);
  else
    return m_internal->m_row_distribution->globalToLocal(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colGlobalToLocal(Integer i, Integer p) const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->globalToLocal(i, p);
  else
    return m_internal->m_row_distribution->globalToLocal(i, p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colLocalToGlobal(Integer i) const
{
  if (m_internal->m_col_distribution.get())
    return m_internal->m_col_distribution->localToGlobal(i);
  else
    return m_internal->m_row_distribution->localToGlobal(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::colLocalToGlobal(Integer i, Integer p) const
{
  if (!m_internal->m_col_distribution.get())
    return m_internal->m_row_distribution->localToGlobal(i, p);
  else
    return m_internal->m_col_distribution->localToGlobal(i, p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
MatrixDistribution::owner(Integer i, Integer j ALIEN_UNUSED_PARAM) const
{
  return m_internal->m_row_distribution->owner(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& nout, const MatrixDistribution& dist)
{
  nout << "row distributed";
  if (dist.isParallel()) {
    nout << ", parallel";
  }
  nout << ", global sizes [col=" << dist.globalColSize()
       << ",row=" << dist.globalRowSize() << "]";
  nout << ", local row size=" << dist.localRowSize();
  nout << ", row offset=" << dist.rowOffset();
  return nout;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
