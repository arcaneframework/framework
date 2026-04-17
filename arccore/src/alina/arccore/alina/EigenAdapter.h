// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EigenAdapter.h                                              (C) 2000-2026 */
/*                                                                           */
/* Adapters for Eigen types to be used with builtin backend.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_EIGENADAPTER_H
#define ARCCORE_ALINA_EIGENADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <type_traits>
#include <Eigen/SparseCore>
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/BuiltinBackend.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//---------------------------------------------------------------------------
// Backend interface specialization for Eigen types
//---------------------------------------------------------------------------
template <class T, class Enable = void>
struct is_eigen_sparse_matrix : std::false_type
{};

template <class T, class Enable = void>
struct is_eigen_type : std::false_type
{};

template <typename Scalar, int Flags, typename Storage>
struct is_eigen_sparse_matrix<Eigen::Map<Eigen::SparseMatrix<Scalar, Flags, Storage>>> : std::true_type
{};

template <typename Scalar, int Flags, typename Storage>
struct is_eigen_sparse_matrix<Eigen::SparseMatrix<Scalar, Flags, Storage>> : std::true_type
{};

template <class T>
struct is_eigen_type<T,
                     typename std::enable_if<std::is_arithmetic<typename T::RealScalar>::value &&
                                             std::is_base_of<Eigen::EigenBase<T>, T>::value>::type> : std::true_type
{};

template <class T>
struct value_type<T, typename std::enable_if<is_eigen_type<T>::value>::type>
{
  typedef typename T::Scalar type;
};

template <class T>
struct nonzeros_impl<T, typename std::enable_if<is_eigen_sparse_matrix<T>::value>::type>
{
  static size_t get(const T& matrix)
  {
    return matrix.nonZeros();
  }
};

template <class T>
struct row_iterator<T, typename std::enable_if<is_eigen_sparse_matrix<T>::value>::type>
{
  typedef typename T::InnerIterator type;
};

template <class T>
struct row_begin_impl<T, typename std::enable_if<is_eigen_sparse_matrix<T>::value>::type>
{
  typedef typename row_iterator<T>::type iterator;
  static iterator get(const T& matrix, size_t row)
  {
    return iterator(matrix, row);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
