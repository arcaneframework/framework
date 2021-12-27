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

/*
 * CSRModifierViewT.h
 *
 *  Created on: Dec 24, 2021
 *      Author: gratienj
 */

// -*- C++ -*-
#pragma once

namespace Alien
{

template <typename ProfileT>
class CSRProfileConstViewT
{
 protected:
  ProfileT const& m_profile;

 public:
  // clang-format off
  typedef ProfileT                        ProfileType ;
  typedef typename ProfileType::IndexType IndexType ;
  // clang-format on

  CSRProfileConstViewT(ProfileT const& profile)
  : m_profile(profile)
  {}

  std::size_t nrows()
  {
    return m_profile.getNRows();
  }

  std::size_t nnz()
  {
    return m_profile.getNnz();
  }

  IndexType const* kcol()
  {
    return m_profile.kcol();
  }

  IndexType const* dcol()
  {
    return m_profile.dcol();
  }

  IndexType const* cols()
  {
    return m_profile.cols();
  }
};

template <typename MatrixT>
class CSRConstViewT
: public CSRProfileConstViewT<typename MatrixT::ProfileType>
{
 public:
  // clang-format off
    typedef MatrixT                           MatrixType ;
    typedef typename MatrixType::ProfileType  ProfileType ;
    typedef CSRProfileConstViewT<ProfileType> BaseType ;
    typedef typename MatrixType::ValueType    ValueType ;
  // clang-format on

  CSRConstViewT(MatrixT const& matrix)
  : BaseType(matrix.getProfile())
  , m_matrix(matrix)
  {}

  ValueType const* data()
  {
    return this->m_matrix.data();
  }

 private:
  MatrixType const& m_matrix;
};

template <typename MatrixT>
class CSRModifierViewT
: public CSRProfileConstViewT<typename MatrixT::ProfileType>
{
 public:
  // clang-format off
    typedef MatrixT                           MatrixType ;
    typedef typename MatrixType::ProfileType  ProfileType ;
    typedef CSRProfileConstViewT<ProfileType> BaseType ;
    typedef typename MatrixType::ValueType    ValueType ;
  // clang-format on

  CSRModifierViewT(MatrixT& matrix)
  : BaseType(matrix.getProfile())
  , m_matrix(matrix)
  {
    m_matrix.notifyChanges();
  }

  virtual ~CSRModifierViewT()
  {
    m_matrix.endUpdate();
  }

  ValueType* data()
  {
    return this->m_matrix.data();
  }

 private:
  MatrixType& m_matrix;
};

} // end namespace Alien
