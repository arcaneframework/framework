// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IO.h                                                        (C) 2000-2026 */
/*                                                                           */
/* Readers for Matrix Market sparse matrices and dense vectors.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_IO_H
#define ARCCORE_ALINA_IO_H
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

#include <climits>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric>

#include <type_traits>
#include <tuple>

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/BackendInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::IO
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Matrix market reader.
class mm_reader
{
 public:

  /// Open the file by name
  explicit mm_reader(const std::string& fname)
  : f(fname)
  {
    precondition(f, "Failed to open file \"" + fname + "\"");

    // Read banner.
    precondition(std::getline(f, line), format_error());

    std::istringstream is(line);
    std::string banner, mtx, coord, dtype, storage;

    precondition(
    is >> banner >> mtx >> coord >> dtype >> storage,
    format_error());

    precondition(banner == "%%MatrixMarket", format_error("no banner"));
    precondition(mtx == "matrix", format_error("not a matrix"));

    if (storage == "general") {
      _symmetric = false;
    }
    else if (storage == "symmetric") {
      _symmetric = true;
    }
    else {
      precondition(false, "unsupported storage type");
    }

    if (coord == "coordinate") {
      _sparse = true;
    }
    else if (coord == "array") {
      _sparse = false;
    }
    else {
      precondition(false, format_error("unsupported coordinate type"));
    }

    if (dtype == "real") {
      _complex = false;
      _integer = false;
    }
    else if (dtype == "complex") {
      _complex = true;
      _integer = false;
    }
    else if (dtype == "integer") {
      _complex = false;
      _integer = true;
    }
    else {
      precondition(false, format_error("unsupported data type"));
    }

    // Skip comments.
    do {
      precondition(std::getline(f, line), format_error("unexpected eof"));
    } while (line[0] == '%');

    // The last line is comment-free and holds the matrix sizes
    is.clear();
    is.str(line);
    precondition(is >> nrows >> ncols, format_error());
  }

  /// Matrix in the file is symmetric.
  bool is_symmetric() const { return _symmetric; }

  /// Matrix in the file is sparse.
  bool is_sparse() const { return _sparse; }

  /// Matrix in the file is complex-valued.
  bool is_complex() const { return _complex; }

  /// Matrix in the file is integer-valued.
  bool is_integer() const { return _integer; }

  /// Number of rows.
  size_t rows() const { return nrows; }

  /// Number of rows.
  size_t cols() const { return ncols; }

  /// Read sparse matrix from the file.
  template <typename Idx, typename Val>
  std::tuple<size_t, size_t> operator()(std::vector<Idx>& ptr,
                                        std::vector<Idx>& col,
                                        std::vector<Val>& val,
                                        ptrdiff_t row_beg = -1,
                                        ptrdiff_t row_end = -1)
  {
    precondition(_sparse, format_error("not a sparse matrix"));
    precondition(Alina::is_complex<Val>::value == _complex,
                 _complex ? "attempt to read complex values into real vector" : "attempt to read real values into complex vector");
    precondition(std::is_integral<Val>::value == _integer,
                 _integer ? "attempt to read integer values into real vector" : "attempt to read real values into integer vector");

    // Read sizes
    ptrdiff_t n, m;
    size_t nnz;
    std::istringstream is;
    {
      // line already holds the matrix sizes
      is.clear();
      is.str(line);
      precondition(is >> n >> m >> nnz, format_error());
    }

    if (row_beg < 0)
      row_beg = 0;
    if (row_end < 0)
      row_end = n;

    precondition(row_beg >= 0 && row_end <= n,
                 "Wrong subset of rows is requested");

    ptrdiff_t _nnz = _symmetric ? 2 * nnz : nnz;

    if (row_beg != 0 || row_end != n)
      _nnz *= 1.2 * (row_end - row_beg) / n;

    std::vector<Idx> _row;
    _row.reserve(_nnz);
    std::vector<Idx> _col;
    _col.reserve(_nnz);
    std::vector<Val> _val;
    _val.reserve(_nnz);

    ptrdiff_t chunk = row_end - row_beg;

    ptr.resize(chunk + 1);
    std::fill(ptr.begin(), ptr.end(), 0);

    for (size_t k = 0; k < nnz; ++k) {
      precondition(std::getline(f, line), format_error("unexpected eof"));
      is.clear();
      is.str(line);

      Idx i, j;
      Val v;

      precondition(is >> i >> j, format_error());

      i -= 1;
      j -= 1;

      v = read_value<Val>(is);

      if (row_beg <= i && i < row_end) {
        ++ptr[i - row_beg + 1];

        _row.push_back(i - row_beg);
        _col.push_back(j);
        _val.push_back(v);
      }

      if (_symmetric && i != j && row_beg <= j && j < row_end) {
        ++ptr[j - row_beg + 1];

        _row.push_back(j - row_beg);
        _col.push_back(i);
        _val.push_back(v);
      }
    }

    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    col.resize(ptr.back());
    val.resize(ptr.back());

    for (size_t k = 0, e = val.size(); k < e; ++k) {
      Idx i = _row[k];
      Idx j = _col[k];
      Val v = _val[k];

      Idx head = ptr[i]++;
      col[head] = j;
      val[head] = v;
    }

    std::rotate(ptr.begin(), ptr.end() - 1, ptr.end());
    ptr.front() = 0;

    arccoreParallelFor(0, chunk, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        Idx beg = ptr[i];
        Idx end = ptr[i + 1];

        Alina::detail::sort_row(&col[0] + beg, &val[0] + beg, end - beg);
      }
    });

    return std::make_tuple(chunk, m);
  }

  /// Read dense array from the file.
  template <typename Val>
  std::tuple<size_t, size_t> operator()(std::vector<Val>& val,
                                        ptrdiff_t row_beg = -1,
                                        ptrdiff_t row_end = -1)
  {
    precondition(!_sparse, format_error("not a dense array"));
    precondition(Alina::is_complex<Val>::value == _complex,
                 _complex ? "attempt to read complex values into real vector" : "attempt to read real values into complex vector");
    precondition(std::is_integral<Val>::value == _integer,
                 _integer ? "attempt to read integer values into real vector" : "attempt to read real values into integer vector");

    // Read sizes
    ptrdiff_t n, m;
    std::istringstream is;
    {
      // line already holds the matrix sizes
      is.clear();
      is.str(line);
      precondition(is >> n >> m, format_error());
    }

    if (row_beg < 0)
      row_beg = 0;
    if (row_end < 0)
      row_end = n;

    precondition(row_beg >= 0 && row_end <= n,
                 "Wrong subset of rows is requested");

    val.resize((row_end - row_beg) * m);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        precondition(std::getline(f, line), format_error("unexpected eof"));
        if (row_beg <= i && i < row_end) {
          is.clear();
          is.str(line);
          val[(i - row_beg) * m + j] = read_value<Val>(is);
        }
      }
    }

    return std::make_tuple(row_end - row_beg, m);
  }

 private:

  std::ifstream f;
  std::string line;

  bool _sparse;
  bool _symmetric;
  bool _complex;
  bool _integer;

  size_t nrows, ncols;

  std::string format_error(const std::string& msg = "") const
  {
    std::string err_string = "MatrixMarket format error";
    if (!msg.empty())
      err_string += " (" + msg + ")";
    return err_string;
  }

  template <typename T>
  typename std::enable_if<Alina::is_complex<T>::value, T>::type
  read_value(std::istream& s)
  {
    typename math::scalar_of<T>::type x, y;
    precondition(s >> x >> y, format_error());
    return T(x, y);
  }

  template <typename T>
  typename std::enable_if<!Alina::is_complex<T>::value, T>::type
  read_value(std::istream& s)
  {
    T x;
    if (std::is_same<T, char>::value) {
      // Special case:
      // We want to read 8bit integers from MatrixMarket, not chars.
      int i;
      precondition(s >> i, format_error());
      x = static_cast<char>(i);
    }
    else {
      precondition(s >> x, format_error());
    }
    return x;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{
  template <typename Val>
  typename std::enable_if<is_complex<Val>::value, std::ostream&>::type
  write_value(std::ostream& s, Val v)
  {
    return s << std::scientific << std::setprecision(20) << std::real(v) << " " << std::imag(v);
  }

  template <typename Val>
  typename std::enable_if<!is_complex<Val>::value, std::ostream&>::type
  write_value(std::ostream& s, Val v)
  {
    return s << std::scientific << std::setprecision(20) << v;
  }

} // namespace detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Write dense array in Matrix Market format.
template <typename Val>
void mm_write(const std::string& fname,
              const Val* data,
              size_t rows,
              size_t cols = 1)
{
  std::ofstream f(fname.c_str());
  precondition(f, "Failed to open file \"" + fname + "\" for writing");

  // Banner
  f << "%%MatrixMarket matrix array ";
  if (is_complex<Val>::value) {
    f << "complex ";
  }
  else if (std::is_integral<Val>::value) {
    f << "integer ";
  }
  else {
    f << "real ";
  }
  f << "general\n";

  // Sizes
  f << rows << " " << cols << "\n";

  // Data
  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      detail::write_value(f, data[i * cols + j]) << "\n";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Write sparse matrix in Matrix Market format.
template <class Matrix>
void mm_write(const std::string& fname, const Matrix& A)
{
  typedef typename backend::value_type<Matrix>::type Val;

  const size_t rows = backend::nbRow(A);
  const size_t cols = backend::nbColumn(A);
  const size_t nnz = backend::nonzeros(A);

  std::ofstream f(fname.c_str());
  precondition(f, "Failed to open file \"" + fname + "\" for writing");

  // Banner
  f << "%%MatrixMarket matrix coordinate ";
  if (is_complex<Val>::value) {
    f << "complex ";
  }
  else if (std::is_integral<Val>::value) {
    f << "integer ";
  }
  else {
    f << "real ";
  }
  f << "general\n";

  // Sizes
  f << rows << " " << cols << " " << nnz << "\n";

  // Data
  for (size_t i = 0; i < rows; ++i) {
    for (auto a = backend::row_begin(A, i); a; ++a) {
      f << i + 1 << " " << a.col() + 1 << " ";
      detail::write_value(f, a.value()) << "\n";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Read single value from a binary file.
template <class T>
bool read(std::ifstream& f, T& val)
{
  return static_cast<bool>(f.read((char*)&val, sizeof(T)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Read vector from a binary file.
template <class T>
bool read(std::ifstream& f, std::vector<T>& vec)
{
  return static_cast<bool>(f.read((char*)&vec[0], sizeof(T) * vec.size()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Get size of the CRS matrix stored in a binary file
template <typename IndexType>
IndexType crs_size(const std::string& fname)
{
  std::ifstream f(fname.c_str(), std::ios::binary);
  IndexType n;

  precondition(f, "Failed to open matrix file");
  precondition(read(f, n), "File I/O error");

  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Read CRS matrix from a binary file.
template <typename SizeT, typename Ptr, typename Col, typename Val>
void read_crs(const std::string& fname,
              SizeT& n,
              std::vector<Ptr>& ptr,
              std::vector<Col>& col,
              std::vector<Val>& val,
              ptrdiff_t row_beg = -1,
              ptrdiff_t row_end = -1)
{
  std::ifstream f(fname.c_str(), std::ios::binary);
  precondition(f, "Failed to open matrix file");

  precondition(read(f, n), "File I/O error");

  if (row_beg < 0)
    row_beg = 0;
  if (row_end < 0)
    row_end = n;

  precondition(row_beg >= 0 && row_end <= static_cast<ptrdiff_t>(n),
               "Wrong subset of rows is requested");

  ptrdiff_t chunk = row_end - row_beg;

  ptr.resize(chunk + 1);

  size_t ptr_beg = sizeof(SizeT);
  f.seekg(ptr_beg + row_beg * sizeof(Ptr));
  precondition(read(f, ptr), "File I/O error");

  Ptr nnz;
  f.seekg(ptr_beg + n * sizeof(Ptr));
  precondition(read(f, nnz), "File I/O error");

  SizeT nnz_beg = ptr.front();
  if (nnz_beg)
    for (auto& p : ptr)
      p -= nnz_beg;

  col.resize(ptr.back());
  val.resize(ptr.back());

  size_t col_beg = ptr_beg + (n + 1) * sizeof(Ptr);
  f.seekg(col_beg + nnz_beg * sizeof(Col));
  precondition(read(f, col), "File I/O error");

  f.seekg(col_beg + nnz * sizeof(Col) + nnz_beg * sizeof(Val));
  precondition(read(f, val), "File I/O error");

  arccoreParallelFor(0, chunk, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      Ptr beg = ptr[i];
      Ptr end = ptr[i + 1];
      Alina::detail::sort_row(&col[beg], &val[beg], end - beg);
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SizeT>
void dense_size(const std::string& fname, SizeT& n, SizeT& m)
{
  std::ifstream f(fname.c_str(), std::ios::binary);
  precondition(f, "Failed to open matrix file");

  precondition(read(f, n), "File I/O error");
  precondition(read(f, m), "File I/O error");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename SizeT, typename Val>
void read_dense(const std::string& fname,
                SizeT& n, SizeT& m, std::vector<Val>& v,
                ptrdiff_t row_beg = -1, ptrdiff_t row_end = -1)
{
  std::ifstream f(fname.c_str(), std::ios::binary);
  precondition(f, "Failed to open matrix file");

  precondition(read(f, n), "File I/O error");
  precondition(read(f, m), "File I/O error");

  if (row_beg < 0)
    row_beg = 0;
  if (row_end < 0)
    row_end = n;

  precondition(row_beg >= 0 && row_end <= static_cast<ptrdiff_t>(n),
               "Wrong subset of rows is requested");

  ptrdiff_t chunk = row_end - row_beg;

  v.resize(chunk * m);

  f.seekg(2 * sizeof(SizeT) + row_beg * m * sizeof(Val));
  precondition(read(f, v), "File I/O error");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Write single value to a binary file.
template <class T>
bool write(std::ofstream& f, const T& val)
{
  return static_cast<bool>(f.write((char*)&val, sizeof(T)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Write vector to a binary file.
template <class T>
bool write(std::ofstream& f, const std::vector<T>& vec)
{
  return static_cast<bool>(f.write((char*)&vec[0], sizeof(T) * vec.size()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::IO

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
