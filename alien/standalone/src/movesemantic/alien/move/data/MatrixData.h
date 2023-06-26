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

#include <memory>

#include <arccore/base/String.h>

#include <alien/data/IMatrix.h>

#include <alien/move/AlienMoveSemanticPrecomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
class IMessagePassingMng;
}

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Block;
class VBlock;
class Space;
class MatrixDistribution;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Move
{

  //! Algebraic Matrix with internal multi-representation object.
  class ALIEN_MOVESEMANTIC_EXPORT MatrixData : public IMatrix
  {
   public:
    typedef Real ValueType;

    /*! @defgroup constructor Matrix Constructor
         * @{
         */
    /*! Empty constructor
         *
         * This matrix must be associated with a \r Space before use.
         */
    MatrixData();

    /*! Build a new matrix from a Space
         *
         * \param space Domain (and Codomain) Space of the matrix.
         * \param dist Parallel distribution.
         *
         * This matrix is directly ready to use. */
    [[deprecated]] MatrixData(const Space& space, const MatrixDistribution& dist);

    /*!  Build a new matrix from two Spaces
         *
         * \param row_space Domain Space of the matrix.
         * \param col_space Codomain Space of the matrix.
         * \param dist Parallel distribution.
         *
         * This matrix is directly ready to use. */
    [[deprecated]] MatrixData(
    const Space& row_space, const Space& col_space, const MatrixDistribution& dist);

    /*! Build a new matrix from a size.
         *
         * Matlab-like interface, matrix is defined as a [0, n-1]x[0, n-1] array.
         * \param size Number of rows and columns of the matrix.
         * \param dist Parallel distribution.
         *
         * This matrix is ready to use on an anonymous Space.
         */
    [[deprecated]] MatrixData(Integer size, const MatrixDistribution& dist);

    /*! Build a new matrix from two sizes.
         *
         * Matlab-like interface, matrix is defined as a [0, n-1]x[0, m-1] array.
         * \param row_size Number of rows of the matrix.
         * \param col_size Number of rows of the matrix.
         * \param dist Parallel distribution.
         *
         * This matrix is ready to use on an anonymous Space.
         */
    [[deprecated]] MatrixData(Integer row_size, Integer col_size, const MatrixDistribution& dist);

    /*!  Build a new matrix on `MatrixDistribution`
      *
      * \param dist Parallel distribution.
      *
      * This matrix is directly ready to use. */
    explicit MatrixData(const MatrixDistribution& dist);

    /*! Move constructor for Matrix
         *
         * @param matrix Matrix to move from.
         */
    MatrixData(MatrixData&& matrix);
    //! }@

    /*! Destructor
         * All internal data structures will be deleted.
         */
    virtual ~MatrixData() = default;

    /*! Move assignment
         * \brief Move from Matrix
         *
         * @param matrix Matrix to move from.
         */
    MatrixData& operator=(MatrixData&& matrix);

    /*! Initialize a Matrix with a Space.
         *
         * @param space Domain and Codomain Space for the Matrix.
         * @param dist Parallel Distribution.
         */
    void init(const Space& space, const MatrixDistribution& dist);

    /*! Only support move semantic */
    MatrixData(const MatrixData&) = delete;
    /*! Only support move semantic */
    void operator=(const MatrixData&) = delete;

    MatrixData clone() const;

    /*! @defgroup block Block related API
         * @{ */

    void setBlockInfos(const Integer block_size);

    void setBlockInfos(const Block* block);

    void setBlockInfos(const VBlock* block);

    Block const* block() const;

    VBlock const* vblock() const;
    /*! }@ */

    /*! Delete all internal data structures */
    void free();

    /*! Clean all internal data structures.
         *
         * Internal data are cleared, not deleted.
         */
    void clear();

    /*! Handle for visitor pattern */
    void visit(ICopyOnWriteMatrix&) const;

    /*! @defgroup space Space related functions.
         * @{
         */
    /*! Domain Space of the current matrix
         * @return Domain Space.
         * @throw FatalException if uninitialized.
         * Call isNull before to avoid any problem.
         */
    const ISpace& rowSpace() const;

    /*! CoDomain Space of the current matrix
         * @return CoDomain Space.
         * @throw FatalException if uninitialized.
         * Call isNull before to avoid any problem.
         */
    const ISpace& colSpace() const;
    /*! }@ */

    /*! Parallel distribution of the Matrix.
         *
         * @return Parallel distribution of the Matrix.
         */
    const MatrixDistribution& distribution() const;

    /*! @defgroup lock Protection functions.
         * @{
         */
    /*! Lock Matrix with the caller. */
    void lock() {}

    /*! Unlock Matrix, making it available for others. */
    void unlock() {}

    /*! Test if a Matrix is locked.
         *
         * @return whether of not a matrix is already locked by someone.
         */
    bool isLocked() const { return false; }
    /*! }@ */

    /*! @defgroup properties Algebraic properties management.
         *
         * Algebraic properties are designed to propagate high level information of matrix
         * object. These properties can be passed to external solvers but are not designed to
         * overload Alien's solver parameters.
         * @{ */
    /*! Add a new property on this matrix */
    void setUserFeature(String feature);

    /*! Check if a property is set. */
    bool hasUserFeature(String feature) const;

    /*! Alias on property "transposed" */
    bool isTransposed() const { return hasUserFeature("transposed"); }

    /*! Is this matrix composite ? */
    bool isComposite() const;
    /* }@ */

    /*! @defgroup impl Internal data structure access.
         *
         * Access multi-representation object.
         * @{
         */
    MultiMatrixImpl* impl();

    const MultiMatrixImpl* impl() const;
    /*! } @ */

    friend MatrixData createMatrixData(std::shared_ptr<MultiMatrixImpl> multi);

   private:
    std::shared_ptr<MultiMatrixImpl> m_impl;
  };

  MatrixData ALIEN_MOVESEMANTIC_EXPORT
  readFromMatrixMarket(Arccore::MessagePassing::IMessagePassingMng* pm, const std::string& filename);

  MatrixData createMatrixData(std::shared_ptr<MultiMatrixImpl> multi);
} // namespace Move
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
