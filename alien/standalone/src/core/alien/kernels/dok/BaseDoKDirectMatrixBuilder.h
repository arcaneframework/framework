/*
 * Copyright 2021 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALIEN_BASEDOKDIRECTMATRIXBUILDER_H
#define ALIEN_BASEDOKDIRECTMATRIXBUILDER_H

#include <alien/utils/Precomp.h>

#include <optional>

namespace Alien
{
// Forward declarations.
class IMatrix;
class DoKMatrix;

namespace Common
{

  /*!
   * Use DoKMatrix to implement DirectMatrixBuilder.
   *
   * This data structure is not compressed like CSR, allowing easy insertion of values.
   */
  class ALIEN_EXPORT BaseDoKDirectMatrixBuilder
  {
   public:
    /*!
     * Build a new representation of a matrix
     * @param self the multi-representation handler. It is locked during the life of this object.
     */
    // FIXME: This should be implemented with move-semantic.
    explicit BaseDoKDirectMatrixBuilder(IMatrix& self);
    ~BaseDoKDirectMatrixBuilder();

    /*!
     * Add a contribution to a non-zero. Will create the non-zero if needed.
     * Non-zero does not have to be local.
     *
     * @param row
     * @param col
     * @param value of the contribution
     * @return a value if insertion is ok
     */
    std::optional<Real> contribute(Integer row, Integer col, Real value);

    /*!
     * Set a value to a non-zero. Will create the non-zero if needed.
     * Non-zero does not have to be local, but behavior for gathering non-local non-zero is undefined.
     * It is not possible to choose between 2 nnz from different processes.
     *
     * @param row
     * @param col
     * @param value of the contribution
     * @return a value if insertion is ok
     */
    std::optional<Real> setNNZ(Integer row, Integer col, Real value);

    /*!
     * Sort and compact data.
     *
     * @return if successful.
     */
    bool assemble();

   private:
    //! Convenience reference to multi-repr manager.
    IMatrix& m_matrix;
    //! Pointer to data storage. Managed by m_matrix.
    DoKMatrix* m_impl;
  };

} // namespace Common
} // namespace Alien
#endif //ALIEN_BASEDOKDIRECTMATRIXBUILDER_H
