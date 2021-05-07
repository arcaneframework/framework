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

#include <alien/data/IVector.h>
#include <alien/move/AlienMoveSemanticPrecomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Block;
class VBlock;
class Space;
class VectorDistribution;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Algebraic Vector with internal multi-representation object.
class ALIEN_MOVESEMANTIC_EXPORT VectorData : public IVector
{
 public:
  typedef Real ValueType;

 public:
  /*! @defgroup constructor Vector Constructor
   * @{
   */
  /*! Empty constructor
   *
   * This vector must be associated with a Space before use.
   */
  VectorData();

  /*! Build a new Vector from a Space
   *
   * \param space Definition Space of the Vector.
   * \param dist Parallel distribution.
   *
   * This vector is directly ready to use. */
  VectorData(const ISpace& space, const VectorDistribution& dist);

  /*! Build a new Vector from a size.
   *
   * Matlab-like interface, Vector is defined as a [0, n-1] array.
   * \param size Number of elements of the vector.
   * \param dist Parallel distribution.
   *
   * This vector is ready to use on an anonymous Space.
   */
  VectorData(Integer size, const VectorDistribution& dist);

  /*! Move constructor for Vector
   *
   * @param vector Vector to move from.
   */
  VectorData(VectorData&& vector);
  /*! }@ */

  /*! Destructor
   * All internal data structures will be deleted.
   */
  virtual ~VectorData();

  /*! Move assignment
   * \brief Move from Vector
   *
   * @param matrix Vector to move from.
   */
  void operator=(VectorData&& vector);

  /*! Initialize a Vector with a Space.
   *
   * @param space Definition Space.
   * @param dist Parallel Distribution.
   */
  void init(const ISpace& space, const VectorDistribution& dist);

 private:
  VectorData(const VectorData&) = delete;
  void operator=(const VectorData&) = delete;

 public:
  /*! @defgroup block Block related API
   * @{ */
  void setBlockInfos(const Integer block_size);
  // void setBlockInfos(const IBlockBuilder& block_size);
  void setBlockInfos(const Block* block);
  void setBlockInfos(const VBlock* block);

  const Block* block() const;
  const VBlock* vblock() const;
  /*! }@ */

  /*! Delete all internal data structures */
  void free();

  /*! Clean all internal data structures.
   *
   * Internal data are cleared, not deleted.
   */
  void clear();

  /*! Handle for visitor pattern */
  void visit(ICopyOnWriteVector&) const;

  /*! @defgroup space Space related functions.
   * @{
   */
  /*! Domain Space of the current vector
   * @return Definition Space.
   * @throw FatalException if uninitialized.
   * Call isNull before to avoid any problem.
   */
  const ISpace& space() const;

  /*! Parallel distribution of the Vector.
   *
   * @return Parallel distribution of the Vector.
   */
  const VectorDistribution& distribution() const;

  /*! @defgroup properties Algebraic properties management.
   *
   * Algebraic properties are designed to propagate high level information of matrix
   * object. These properties can be passed to external solvers but are not designed to
   * overload Alien's solver parameters.
   * @{ */
  /*! Add a new property on this vector */
  void setUserFeature(String feature);
  /*! Check if a property is set. */
  bool hasUserFeature(String feature) const;
  /*! }@ */

 public:
  /*! @defgroup impl Internal data structure access.
   *
   * Access multi-representation object.
   * @{
   */
  MultiVectorImpl* impl();
  const MultiVectorImpl* impl() const;
  /*! }@ */

 private:
  std::shared_ptr<MultiVectorImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
