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

#include "DoKDistributor.h"

#include <vector>

#include "DoKBackEnd.h"
#include "DoKMatrixT.h"
#include "DoKVector.h"

#include <alien/kernels/redistributor/RedistributorCommPlan.h>

namespace Alien
{

DoKDistributor::DoKDistributor(const RedistributorCommPlan* commPlan)
: m_distributor(new DoKDistributorComm(commPlan))
{}

void DoKDistributor::distribute(const DoKMatrix& src, DoKMatrix& dst)
{
  distribute(src.data(), dst.data());
}

void DoKDistributor::distribute(const DoKVector& src, DoKVector& dst)
{
  std::vector<std::pair<Arccore::Int32, DoKVector::ValueType>> pair_values(src.m_data.begin(), src.m_data.end());
  std::sort(pair_values.begin(), pair_values.end());

  Arccore::UniqueArray<Int32> snd_keys(src.m_data.size());
  Arccore::UniqueArray<DoKVector::ValueType> snd_values(src.m_data.size());

  int i = 0;
  for (auto k : pair_values) {
    snd_keys[i] = k.first;
    snd_values[i] = k.second;
    i++;
  }

  m_distributor->computeCommPlan(snd_keys);

  const auto size = m_distributor->rcvSize();
  // We split in 2 arrays to be able to use Arccore ...
  UniqueArray<DoKVector::ValueType> rcv_values(size);
  UniqueArray<Int32> rcv_keys(size);
  m_distributor->exchange(snd_keys.constView(), rcv_keys.view());
  m_distributor->exchange(snd_values.constView(), rcv_values.view());

  dst.m_data.clear();

  for (int offset = 0; offset < size; ++offset) {
    dst.set(rcv_keys[offset], rcv_values[offset]);
  }
}

} // namespace Alien
