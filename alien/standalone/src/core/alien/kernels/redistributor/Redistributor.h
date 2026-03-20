// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

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
  enum class Method { dok, csr };

  Redistributor(int globalSize, IMessagePassingMng* super, IMessagePassingMng* target, Method method = Method::dok);
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
  Method m_method = Method::dok;
};

} // namespace Alien
