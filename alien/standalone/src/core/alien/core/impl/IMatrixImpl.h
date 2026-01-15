// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/*!
 * \file IMatrixImpl.h
 * \brief IMatrixImpl.h
 */

#pragma once

#include <arccore/base/String.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/time_stamp/Timestamp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

typedef String BackEndId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiMatrixImpl;
class ISpace;
class Block;
class VBlock;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup impl
 * \brief Interface to handle abstract matrices implementation
 */
class ALIEN_EXPORT IMatrixImpl : public Timestamp
, public ObjectWithTrace
{
 public:
  //! Type of the values stored in the matrix
  typedef Arccore::Real ValueType;

 public:
  /*!
   * \brief Constructor
   * \param[in] multi_impl Pointer to the multimatrix handler, null if not associated
   * \param[in] backend Name of the underneath backend, empty string if not associated
   */
  explicit IMatrixImpl(const MultiMatrixImpl* multi_impl, BackEndId backend = "");

  //! Free resources
  virtual ~IMatrixImpl() override = default;

  IMatrixImpl(const IMatrixImpl& src) = delete;
  IMatrixImpl(IMatrixImpl&& src) = delete;
  IMatrixImpl& operator=(const IMatrixImpl& src) = delete;
  IMatrixImpl& operator=(IMatrixImpl&& src) = delete;

 public:
  //! Wipe out internal data.
  virtual void clear() {}

  /*!
   * \brief Get the row space associated to the matrix
   *\ returns The row space
   */
  virtual const ISpace& rowSpace() const;

  /*!
   * \brief Get the row space associated to the matrix
   *\ returns The row space
   */
  virtual const ISpace& colSpace() const;

  /*!
   * \brief Get the distribution of the matrix
   * \returns The matrix distribution
   */
  virtual const MatrixDistribution& distribution() const;

  /*!
   * Get the matrix backend id
   * \returns The backend if associated to the matrix
   */
  virtual BackEndId backend() const { return m_backend; }

  /*!
   * \brief Get block datas of the matrix
   *
   * Get the block datas of the matrix. This method should be used only with uniform
   * blocks matrices, otherwise il will return a nullptr
   *
   * \returns Block data for block matrices, nullptr for scalar and variable block
   * matrices
   */
  virtual const Block* block() const;

  /*!
   * \brief Get block datas of the matrix
   *
   * Get the block datas of the matrix. This method should be used only with variable
   * blocks matrices, otherwise il will return a nullptr
   *
   * \returns Block data for variable block matrices, nullptr for scalar and block
   * matrices
   */
  virtual const VBlock* vblock() const;

  /*!
   * \brief Get row block datas of the matrix
   *
   * Get the row block datas of the matrix. This method should be used only with variable
   * blocks matrices, otherwise il will return a nullptr
   *
   * \returns Rows block data for variable block matrices, nullptr for scalar and block
   * matrices
   */
  virtual const VBlock* rowBlock() const;

  /*!
   * \brief Get col block datas of the matrix
   *
   * Get the col block datas of the matrix. This method should be used only with variable
   * blocks matrices, otherwise il will return a nullptr
   *
   * \returns Cols block data for variable block matrices, nullptr for scalar and block
   * matrices
   */
  virtual const VBlock* colBlock() const;

  const MultiMatrixImpl* impls() const {
    return m_multi_impl;
  }
 protected:
  //! Pointer on matrices implementation
  const MultiMatrixImpl* m_multi_impl;
  //! Backend id
  BackEndId m_backend;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
