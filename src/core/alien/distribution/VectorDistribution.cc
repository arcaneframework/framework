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
 * \file VectorDistribution.cc
 * \brief VectorDistribution.cc
 */

#include "VectorDistribution.h"

#include <iostream>
#include <numeric>

#include <arccore/message_passing/Messages.h>

#include <alien/data/Space.h>
#include <alien/utils/Trace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal implementation of the vector distribution
struct VectorDistribution::Internal
{
  //! Constructor
  Internal();

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& space, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& space, std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_size, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_size, std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& space, Integer local_size, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] space The space of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(const ISpace& space, Integer local_size,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_size, Integer local_size, IMessagePassingMng* parallel_mng);

  /*!
   * \brief Constructor
   * \param[in] global_size The global size of the vector
   * \param[in] local_size The local size of the vector
   * \param[in] parallel_mng The parallel manager
   */
  Internal(Integer global_size, Integer local_size,
           std::shared_ptr<IMessagePassingMng> parallel_mng);

  //! Free resources
  ~Internal() = default;

  /*!
   * \brief Comparison operator
   * \param[in] dist The distribution to compare
   * \returns Whether or not the distribution are identical
   */
  bool operator==(const Internal& dist) const;

  /*!
   * \brief Whether or not the run is parallel
   * \returns Whether or not the run is parallel
   */
  bool isParallel() const;

  /*!
   * \brief Get the owner of an entry
   * \param[in] i The global id of the element
   * \returns The proc owner
   */
  Integer owner(Integer i) const;

  //! The space
  std::shared_ptr<ISpace> m_space;

  // Is shared_ptr necessary ?
  //! The parallel manager
  std::shared_ptr<IMessagePassingMng> m_parallel_mng;

  //! The local rank
  Integer m_rank;
  //! The global size
  Integer m_global_size;
  //! The local size
  Integer m_local_size;
  //! The local offset
  Integer m_offset;
  //! The array of offsets for all proc
  UniqueArray<Integer> m_offsets;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal()
: m_parallel_mng(nullptr)
, m_rank(0)
, m_global_size(0)
, m_local_size(0)
, m_offset(0)
, m_offsets(1, 0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
const ISpace& space, IMessagePassingMng* parallel_mng)
: m_space(space.clone())
, m_parallel_mng(std::shared_ptr<IMessagePassingMng>(
  parallel_mng, [](IMessagePassingMng*) {})) // Do not call delete.
, m_rank(0)
, m_global_size(m_space->size())
, m_local_size(0)
, m_offset(0)
{
  if (!parallel_mng)
    return;
  m_rank = m_parallel_mng->commRank();
  Integer np = parallel_mng->commSize();
  UniqueArray<Integer> lsize(np, m_global_size / np);
  for (Integer i = 0; i < m_global_size % np; ++i)
    lsize[i]++;
  m_offsets.resize(np, 0);
  for (Integer i = 1; i < np; ++i) {
    m_offsets[i] = lsize[i - 1] + m_offsets[i - 1];
  }
  m_offsets.add(m_global_size);
  m_offset = m_offsets[m_rank];
  m_local_size = lsize[m_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
const ISpace& space, std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_space(space.clone())
, m_parallel_mng(parallel_mng)
, m_rank(0)
, m_global_size(m_space->size())
, m_local_size(0)
, m_offset(0)
{
  if (m_parallel_mng == nullptr)
    return;
  m_rank = m_parallel_mng->commRank();
  Integer np = parallel_mng->commSize();
  UniqueArray<Integer> lsize(np, m_global_size / np);
  for (Integer i = 0; i < m_global_size % np; ++i)
    lsize[i]++;
  m_offsets.resize(np, 0);
  for (Integer i = 1; i < np; ++i) {
    m_offsets[i] = lsize[i - 1] + m_offsets[i - 1];
  }
  m_offsets.add(m_global_size);
  m_offset = m_offsets[m_rank];
  m_local_size = lsize[m_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
Integer global_size, IMessagePassingMng* parallel_mng)
: Internal(Space(global_size), parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
Integer global_size, std::shared_ptr<IMessagePassingMng> parallel_mng)
: Internal(Space(global_size), parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
const ISpace& space, Integer local_size, IMessagePassingMng* parallel_mng)
: m_space(space.clone())
, m_parallel_mng(std::shared_ptr<IMessagePassingMng>(
  parallel_mng, [](IMessagePassingMng*) {})) // Do not use delete
, m_rank(0)
, m_global_size(m_space->size())
, m_local_size(local_size)
, m_offset(0)
{
  if (!parallel_mng)
    return;
  m_rank = m_parallel_mng->commRank();
  Integer np = parallel_mng->commSize();
  UniqueArray<Integer> lsize(np, 0);
  lsize[m_rank] = m_local_size;
  Arccore::MessagePassing::mpAllReduce(
  m_parallel_mng.get(), Arccore::MessagePassing::ReduceMax, lsize.view());
  Integer gsize = std::accumulate(lsize.begin(), lsize.end(), 0);
  if (gsize != m_global_size) {
    alien_fatal([&] {
      cout() << "error, expected global size (" << m_global_size
             << ") is not equal to computed global size (" << gsize << ")";
    });
  }
  m_offsets.resize(np, 0);
  auto tmp = 0;
  for (Integer i = 1; i < np; ++i) {
    tmp += lsize[i - 1];
    m_offsets[i] = tmp;
  }
  m_offsets.add(m_global_size);
  m_offset = m_offsets[m_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(const ISpace& space, Integer local_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_space(space.clone())
, m_parallel_mng(parallel_mng)
, m_rank(0)
, m_global_size(m_space->size())
, m_local_size(local_size)
, m_offset(0)
{
  if (!parallel_mng)
    return;
  m_rank = m_parallel_mng->commRank();
  Integer np = parallel_mng->commSize();
  UniqueArray<Integer> lsize(np, 0);
  lsize[m_rank] = m_local_size;
  Arccore::MessagePassing::mpAllReduce(
  m_parallel_mng.get(), Arccore::MessagePassing::ReduceMax, lsize.view());
  Integer gsize = std::accumulate(lsize.begin(), lsize.end(), 0);
  if (gsize != m_global_size) {
    alien_fatal([&] {
      cout() << "error, expected global size (" << m_global_size
             << ") is not equal to computed global size (" << gsize << ")";
    });
  }
  m_offsets.resize(np, 0);
  auto tmp = 0;
  for (Integer i = 1; i < np; ++i) {
    tmp += lsize[i - 1];
    m_offsets[i] = tmp;
  }
  m_offsets.add(m_global_size);
  m_offset = m_offsets[m_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(
Integer global_size, Integer local_size, IMessagePassingMng* parallel_mng)
: Internal(Space(global_size), local_size, parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::Internal::Internal(Integer global_size, Integer local_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: Internal(Space(global_size), local_size, parallel_mng)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VectorDistribution::Internal::operator==(const Internal& dist) const
{
  return ((dist.m_parallel_mng == m_parallel_mng) && (dist.m_rank == m_rank) && (dist.m_global_size == m_global_size) && (dist.m_local_size == m_local_size) && (dist.m_offset == m_offset));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VectorDistribution::Internal::isParallel() const
{
  return (m_parallel_mng != nullptr) && (m_parallel_mng->commSize() > 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::Internal::owner(Integer i) const
{
  // TODO: use a dichotomous search to accelerate search
  // on the proc
  if (i >= m_offset && i < m_offsets[m_rank + 1]) {
    return m_rank;
  }
  // else on proc > rank
  if (i < m_offset) {
    for (int rk = m_rank - 1; rk > 0; --rk) {
      if (i >= m_offsets[rk]) {
        return rk;
      }
    }
    return 0;
  }
  else { // or on proc < rank
    Integer np = m_parallel_mng->commSize();
    for (int rk = m_rank + 1; rk < np - 1; ++rk) {
      if (i < m_offsets[rk + 1]) {
        return rk;
      }
    }
    return np - 1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution()
: m_internal(std::make_shared<Internal>())
{
  alien_debug([&] { cout() << "Create Empty VectorDistribution"; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(
const ISpace& space, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(space, parallel_mng))
{
  alien_debug(
  [&] { cout() << "Create VectorDistribution(global=" << space.size() << ")"; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(
const ISpace& space, std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(space, parallel_mng))
{
  alien_debug(
  [&] { cout() << "Create VectorDistribution(global=" << space.size() << ")"; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(
Integer global_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(global_size, parallel_mng))
{
  alien_debug(
  [&] { cout() << "Create VectorDistribution(global=" << global_size << ")"; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(
const ISpace& space, Integer local_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(space, local_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create VectorDistribution(global=" << space.size()
           << "local=" << local_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(const ISpace& space, Integer local_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(space, local_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create VectorDistribution(global=" << space.size()
           << "local=" << local_size << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(
Integer global_size, Integer local_size, IMessagePassingMng* parallel_mng)
: m_internal(std::make_shared<Internal>(global_size, local_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create VectorDistribution(global=" << global_size << "local=" << local_size
           << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(Integer global_size, Integer local_size,
                                       std::shared_ptr<IMessagePassingMng> parallel_mng)
: m_internal(std::make_shared<Internal>(global_size, local_size, parallel_mng))
{
  alien_debug([&] {
    cout() << "Create VectorDistribution(global=" << global_size << "local=" << local_size
           << ")";
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(const VectorDistribution& dist)
: m_internal(dist.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::VectorDistribution(VectorDistribution&& dist)
: m_internal(std::move(dist.m_internal))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution::~VectorDistribution() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISpace&
VectorDistribution::space() const
{
  return *m_internal->m_space;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution&
VectorDistribution::operator=(const VectorDistribution& dist)
{
  m_internal = dist.m_internal;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VectorDistribution&
VectorDistribution::operator=(VectorDistribution&& dist)
{
  m_internal = std::move(dist.m_internal);
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VectorDistribution::operator==(const VectorDistribution& dist) const
{
  return *m_internal == *dist.m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VectorDistribution::isParallel() const
{
  return m_internal->isParallel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::owner(Integer i) const
{
  return m_internal->owner(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<IMessagePassingMng>
VectorDistribution::sharedParallelMng() const
{
  return m_internal->m_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng*
VectorDistribution::parallelMng() const
{
  return m_internal->m_parallel_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::localSize() const
{
  return m_internal->m_local_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::globalSize() const
{
  return m_internal->m_global_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::offset() const
{
  return m_internal->m_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::offset(Integer p) const
{
  return m_internal->m_offsets[p];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
VectorDistribution::offsets() const
{
  return m_internal->m_offsets.constView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<VectorDistribution>
VectorDistribution::clone() const
{
  return std::make_shared<VectorDistribution>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::globalToLocal(Integer i) const
{
  return i - m_internal->m_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::globalToLocal(Integer i, Integer p) const
{
  return i - m_internal->m_offsets[p];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::localToGlobal(Integer i) const
{
  return i + m_internal->m_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VectorDistribution::localToGlobal(Integer i, Integer p) const
{
  return i + m_internal->m_offsets[p];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& nout, const VectorDistribution& dist)
{
  nout << "1d distributed";
  if (dist.isParallel()) {
    nout << ", parallel";
  }
  nout << ", global size=" << dist.globalSize();
  nout << ", local size=" << dist.localSize();
  nout << ", offset=" << dist.offset();
  return nout;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
