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

#include <alien/kernels/redistributor/RedistributorCommPlan.h>
#include <alien/utils/Precomp.h>
#include <memory>

namespace Alien
{

class MultiMatrixImpl;
class MultiVectorImpl;

/**
 * @brief Change MultiObj current representation to another communicator.
 *
 * This object is used to defined the input communicator and create another, included,
 * communicator, depending on the user wish to keep or not the current process.
 * It also provides functions to convert Matrix and Vector from their original
 * communicator (input) to the target communicator.
 * And the other way as well for Vectors.
 *
 */
class ALIEN_EXPORT Redistributor
{
 public:
  using Method = enum { dok,
                        csr };

  Redistributor(int globalSize, IMessagePassingMng* super, IMessagePassingMng* target, Method method = dok);
  virtual ~Redistributor() = default;

  /**
   * @brief Convert a Matrix from its communicator to the target communicator.
   * Matrix initial communicator must be the same than the one used when creating
   * the Redistributor object.
   */
  std::shared_ptr<MultiMatrixImpl> redistribute(MultiMatrixImpl* mat);

  /**
   * @brief Convert a Vector from its communicator to the target communicator.
   * Vector initial communicator must be the same than the one used when creating
   * the Redistributor object.
   */
  std::shared_ptr<MultiVectorImpl> redistribute(MultiVectorImpl* vect);

  /**
   * @brief Convert back a Vector : from the target to its original communicator.
   * Vector original communicator must be the same than the one used when creating
   * the Redistributor object.
   */
  void redistributeBack(MultiVectorImpl* vect);

  const RedistributorCommPlan* commPlan() const;

 private:
  IMessagePassingMng* m_super_pm;
  std::unique_ptr<RedistributorCommPlan> m_distributor;
  Method m_method = dok;
};

} // namespace Alien
