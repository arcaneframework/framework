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

template <typename ProfileT, typename DistProfileInfoT>
class CSRProfileConstViewT
{
 protected:
  // clang-format off
  ProfileT const&         m_profile;
  DistProfileInfoT const& m_dist_info ;
  bool                    m_is_parallel = false;
  // clang-format on

 public:
  // clang-format off
  typedef ProfileT                        ProfileType ;
  typedef typename ProfileType::IndexType IndexType ;
  typedef DistProfileInfoT                DistInfoType ;
  // clang-format on

  CSRProfileConstViewT(ProfileT const& profile,
                       DistInfoType const& dist_info,
                       bool is_parallel = false)
  : m_profile(profile)
  , m_dist_info(dist_info)
  , m_is_parallel(is_parallel)
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

  IndexType const* cols()
  {
    if (m_is_parallel)
      return m_dist_info.m_cols.data();
    else
      return m_profile.cols();
  }

  IndexType const* dcol()
  {
    if (m_is_parallel)
      return m_dist_info.dcol(m_profile);
    else
      return m_profile.dcol();
  }
};

template <typename MatrixT>
class CSRConstViewT
: public CSRProfileConstViewT<typename MatrixT::ProfileType,
                              typename MatrixT::DistStructInfo>
{
 public:
  // clang-format off
    typedef MatrixT                                  MatrixType ;
    typedef typename MatrixType::ProfileType         ProfileType ;
    typedef typename MatrixType::DistStructInfo      DistStructInfo ;
    typedef
    CSRProfileConstViewT<ProfileType,DistStructInfo> BaseType ;
    typedef typename MatrixType::ValueType           ValueType ;
    typedef typename BaseType::IndexType             IndexType ;
  // clang-format on

  CSRConstViewT(MatrixT const& matrix)
  : BaseType(matrix.getProfile(), matrix.getDistStructInfo(), matrix.isParallel())
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
: public CSRProfileConstViewT<typename MatrixT::ProfileType,
                              typename MatrixT::DistStructInfo>
{
 public:
  // clang-format off
    typedef MatrixT                                  MatrixType ;
    typedef typename MatrixType::ProfileType         ProfileType ;
    typedef typename MatrixType::DistStructInfo      DistStructInfo ;
    typedef
    CSRProfileConstViewT<ProfileType,DistStructInfo> BaseType ;
    typedef typename MatrixType::ValueType           ValueType ;
    typedef typename BaseType::IndexType             IndexType ;
  // clang-format on

  CSRModifierViewT(MatrixT& matrix)
  : BaseType(matrix.getProfile(), matrix.getDistStructInfo(), matrix.isParallel())
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
