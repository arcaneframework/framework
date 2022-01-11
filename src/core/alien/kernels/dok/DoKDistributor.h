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

#include <alien/utils/Precomp.h>
#include <alien/distribution/MatrixDistribution.h>

#include <alien/kernels/dok/DoKDistributorComm.h>
#include <alien/kernels/dok/DoKLocalMatrixT.h>

namespace Alien
{

class DoKMatrix;
class DoKVector;
class RedistributorCommPlan;

class ALIEN_EXPORT DoKDistributor
{
 public:
  explicit DoKDistributor(const RedistributorCommPlan* commPlan);
  virtual ~DoKDistributor() = default;

  void distribute(const DoKMatrix& src, DoKMatrix& dst);

  void distribute(const DoKVector& src, DoKVector& dst);

  template <typename NNZValue>
  void distribute(DoKLocalMatrixT<NNZValue>& src, DoKLocalMatrixT<NNZValue>& dst)
  {
    /* Distribution algorithm:
     * 1 - Compact local data
     * 2 - Compute communication plan
     * 3 - Perform communication
     * 4 - Return new local DoKMatrix
     *
     * Note: this function can almost be used to redistribute data accross
     * different IMessagePassingMng.
     */

    src.compact();

    // Copy values to a send buffer, in case src and dst are the same matrix.
    UniqueArray<NNZValue> snd_values = src.getValues(); // TODO: avoid this copy

    m_distributor->computeCommPlan(src.getReverseIndexer());
    UniqueArray<NNZValue> rcv_values(m_distributor->rcvSize());
    m_distributor->exchange(snd_values.constView(), rcv_values.view());

    dst.setMaxNnz(rcv_values.size());

    for (int offset = 0; offset < (int)rcv_values.size(); ++offset) {
      auto index = m_distributor->getCoordinates(offset);
      dst.set(index.first, index.second, rcv_values[offset]);
    }
  }

 private:
  std::unique_ptr<DoKDistributorComm> m_distributor;
};

} // namespace Alien
