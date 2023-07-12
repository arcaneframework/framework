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
 * MatrixWriter.cpp
 *
 *  Created on: Apr 7, 2015
 *      Author: gratienj
 */

#include <alien/ref/import_export/SystemWriter.h>

#include <sstream>
#include <string>
#include <vector>

#ifdef ALIEN_USE_HDF5
#include <hdf5.h>
#endif

#ifdef ALIEN_USE_LIBXML2
#include <libxml/parser.h>
#include <libxml/xmlmemory.h>
#endif

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/core/impl/MultiVectorImpl.h>

#include "HDF5Tools.h"
#include <alien/ref/data/block/BlockMatrix.h>
#include <alien/ref/data/block/BlockVector.h>
#include <alien/ref/data/scalar/Matrix.h>
#include <alien/ref/data/scalar/Vector.h>
#include <alien/ref/import_export/SystemInfo.h>
#include <alien/ref/import_export/SystemReader.h>

namespace Alien
{

SystemWriter::SystemWriter(
std::string const& filename, std::string format, IMessagePassingMng* parallel_mng)
: m_filename(filename)
, m_format(format)
, m_prec(6)
, m_parallel_mng(parallel_mng)
{
  m_rank = 0;
  m_nproc = 1;
  if (m_parallel_mng) {
    m_rank = m_parallel_mng->commRank();
    m_nproc = m_parallel_mng->commSize();
    std::stringstream suf;
    suf << "-R" << m_rank << "P" << m_nproc;
    m_filename = filename + suf.str();
  }
  else {
    m_filename = filename;
  }

  // N2_ABS_RES = ||Ax-b||2
  // N2_RELATIVE2RHS_RES = ||Ax-b||2 / ||b||2
  // N2_RELATIVE2X0_RES = ||Ax-b||2 / ||x0||2
  m_conv_crit_to_str[SolutionInfo::N2_ABS_RES] = std::string("||Ax-b||2");
  m_conv_crit_to_str[SolutionInfo::N2_RELATIVE2RHS_RES] =
  std::string("||Ax-b||2 / ||b||2");
  m_conv_crit_to_str[SolutionInfo::N2_RELATIVE2X0_RES] =
  std::string("||Ax-b||2 / ||x0||2");
}

SystemWriter::~SystemWriter() {}

template <typename FileNodeT>
void SystemWriter::_writeMatrixInfo(Exporter& exporter, FileNodeT& parent_node, int nrows,
                                    int ncols, int nnz, int blk_size, int blk_size2)
{
  FileNodeT matrix_info_node = exporter.createFileNode(parent_node, "matrix-info");

  exporter.write(matrix_info_node, "nrows", nrows);
  exporter.write(matrix_info_node, "ncols", ncols);
  exporter.write(matrix_info_node, "nnz", nnz);
  {
    std::vector<int>& i32buffer = exporter.i32buffer;
    i32buffer.reserve(2);
    i32buffer.push_back(blk_size);
    i32buffer.push_back(blk_size2);
    exporter.write(matrix_info_node, "blk-size", i32buffer);
  }

  exporter.closeFileNode(matrix_info_node);
}

template <typename FileNodeT>
void SystemWriter::_writeCSRProfile(Exporter& exporter, FileNodeT& parent_node, int nrows,
                                    int nnz, int const* kcol, int const* cols)
{
  std::vector<int>& i32buffer = exporter.i32buffer;
  FileNodeT profile_node = exporter.createFileNode(parent_node, "profile");
  exporter.write(profile_node, "profile-type", std::string("CSR"));

  exporter.write(profile_node, "nrows", nrows);
  exporter.write(profile_node, "nnz", nnz);

  {
    i32buffer.reserve(nrows + 1);
    for (Integer i = 0; i < nrows + 1; ++i)
      i32buffer.push_back(kcol[i]);
    exporter.write(profile_node, "kcol", i32buffer);
  }
  {
    i32buffer.reserve(nnz);
    for (Integer i = 0; i < nnz; ++i)
      i32buffer.push_back(cols[i]);
    exporter.write(profile_node, "cols", i32buffer);
  }
  exporter.closeFileNode(profile_node);
}

template <typename FileNodeT>
void SystemWriter::_writeMatrixValues(Exporter& exporter, FileNodeT& parent_node, int nnz,
                                      int blk_size, int blk_size2, double const* values)
{
  std::vector<double>& rbuffer = exporter.rbuffer;
  std::vector<int>& i32buffer = exporter.i32buffer;
  FileNodeT data_node = exporter.createFileNode(parent_node, "data");
  {
    {
      FileNodeT info_node = exporter.createFileNode(data_node, "struct-info");

      exporter.write(info_node, "size", nnz);

      {
        i32buffer.reserve(2);
        i32buffer.push_back(blk_size);
        i32buffer.push_back(blk_size2);
        exporter.write(info_node, "blk-size", i32buffer);
      }
      exporter.closeFileNode(info_node);
    }
    rbuffer.reserve(nnz * blk_size * blk_size2);
    for (Integer i = 0; i < nnz * blk_size * blk_size2; ++i)
      rbuffer.push_back(values[i]);
    exporter.write(data_node, "values", rbuffer);
  }
  exporter.closeFileNode(data_node);
}

template <typename FileNodeT>
void SystemWriter::_writeVector(Exporter& exporter, FileNodeT& parent_node, int nrows,
                                int blk_size, double const* values)
{
  std::vector<double>& rbuffer = exporter.rbuffer;
  FileNodeT data_node = exporter.createFileNode(parent_node, "data");
  {
    FileNodeT info_node = exporter.createFileNode(data_node, "struct-info");

    exporter.write(info_node, "size", nrows);
    exporter.write(info_node, "blk-size", blk_size);

    exporter.closeFileNode(info_node);
  }
  {
    rbuffer.reserve(nrows * blk_size);
    for (Integer i = 0; i < nrows * blk_size; ++i)
      rbuffer.push_back(values[i]);
    exporter.write(data_node, "values", rbuffer);
  }
  exporter.closeFileNode(data_node);
}

template <typename FileNodeT>
void SystemWriter::_writeSolutionInfo(
Exporter& exporter, FileNodeT& parent_node, const SolutionInfo& sol_info)
{

  FileNodeT solution_info_node = exporter.createFileNode(parent_node, "solution_info");

  exporter.write(parent_node, "conv-crit", m_conv_crit_to_str[sol_info.m_conv_crit]);
  exporter.write(parent_node, "conv-crit-value", sol_info.m_conv_crit_value);
  exporter.write(parent_node, "solver-comment", sol_info.m_solver_comment);

  exporter.closeFileNode(solution_info_node);
}

template <typename FileNodeT>
void SystemWriter::_beginDump(Exporter*& exporter, FileNodeT& base_node)
{
  exporter = new Exporter(m_filename, m_format, m_prec);
  base_node = exporter->createFileNode("system");
}

template <typename FileNodeT>
void SystemWriter::_endDump(Exporter* exporter, FileNodeT& base_node)
{
  exporter->closeFileNode(base_node);
  delete exporter;
}

void SystemWriter::dump(Matrix const& A)
{
  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int blk_size = 1;
  int nnz = profile.getNnz();
  typedef Exporter::FileNode FileNode;
  Exporter* exporter;
  FileNode root_node;

  _beginDump(exporter, root_node);

  FileNode matrix_node = exporter->createFileNode(root_node, "matrix");
  {
    _writeMatrixInfo(*exporter, matrix_node, nrows, nrows, nnz, blk_size, blk_size);

    const int* cols = profile.cols();
    const int* kcol = profile.kcol();
    _writeCSRProfile(*exporter, matrix_node, nrows, nnz, kcol, cols);

    const double* values = csr.getAddressData();
    _writeMatrixValues(*exporter, matrix_node, nnz, blk_size, blk_size, values);
  }
  exporter->closeFileNode(matrix_node);

  _endDump(exporter, root_node);
}

void SystemWriter::dump(Matrix const& A, Vector const& rhs)
{
  // A is supposed to be square

  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int blk_size = 1;
  int nnz = profile.getNnz();
  typedef Exporter::FileNode FileNode;
  Exporter* exporter;
  FileNode root_node;

  _beginDump(exporter, root_node);

  FileNode matrix_node = exporter->createFileNode(root_node, "matrix");
  {
    _writeMatrixInfo(*exporter, matrix_node, nrows, nrows, nnz, blk_size, blk_size);

    const int* cols = profile.cols();
    const int* kcol = profile.kcol();
    _writeCSRProfile(*exporter, matrix_node, nrows, nnz, kcol, cols);

    const double* values = csr.getAddressData();
    _writeMatrixValues(*exporter, matrix_node, nnz, blk_size, blk_size, values);
  }
  exporter->closeFileNode(matrix_node);

  FileNode vector_node = exporter->createFileNode(root_node, "rhs-0");
  {
    const SimpleCSRVector<Real>& v = rhs.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, vector_node, nrows, blk_size, values);
  }
  exporter->closeFileNode(vector_node);

  _endDump(exporter, root_node);
}

void SystemWriter::dump(
Matrix const& A, Vector const& rhs, Vector const& sol, SolutionInfo const& sol_info)
{
  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int blk_size = 1;
  int nnz = profile.getNnz();
  typedef Exporter::FileNode FileNode;
  Exporter* exporter;
  FileNode root_node;

  _beginDump(exporter, root_node);

  FileNode matrix_node = exporter->createFileNode(root_node, "matrix");
  {
    _writeMatrixInfo(*exporter, matrix_node, nrows, nrows, nnz, blk_size, blk_size);

    const int* cols = profile.cols();
    const int* kcol = profile.kcol();
    _writeCSRProfile(*exporter, matrix_node, nrows, nnz, kcol, cols);

    const double* values = csr.getAddressData();
    _writeMatrixValues(*exporter, matrix_node, nnz, blk_size, blk_size, values);
  }
  exporter->closeFileNode(matrix_node);

  FileNode rhs_node = exporter->createFileNode(root_node, "rhs-0");
  {
    const SimpleCSRVector<Real>& v = rhs.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, rhs_node, nrows, blk_size, values);
  }
  exporter->closeFileNode(rhs_node);

  FileNode solution_node = exporter->createFileNode(root_node, "solution-0");
  {

    _writeSolutionInfo(*exporter, solution_node, sol_info);

    exporter->write(solution_node, "rhs-ref", std::string("rhs-0"));

    FileNode solution_vector_node =
    exporter->createFileNode(solution_node, "solution-vector");
    const SimpleCSRVector<Real>& v = sol.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, solution_vector_node, nrows, blk_size, values);
    exporter->closeFileNode(solution_vector_node);
  }
  exporter->closeFileNode(solution_node);

  _endDump(exporter, root_node);
}

void SystemWriter::dump(BlockMatrix const& A, BlockVector const& rhs)
{
  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int blk_size = A.block().size();
  int nnz = profile.getNnz();
  typedef Exporter::FileNode FileNode;
  Exporter* exporter;
  FileNode root_node;

  _beginDump(exporter, root_node);

  FileNode matrix_node = exporter->createFileNode(root_node, "matrix");
  {
    _writeMatrixInfo(*exporter, matrix_node, nrows, nrows, nnz, blk_size, blk_size);

    const int* cols = profile.cols();
    const int* kcol = profile.kcol();
    _writeCSRProfile(*exporter, matrix_node, nrows, nnz, kcol, cols);

    const double* values = csr.getAddressData();
    _writeMatrixValues(*exporter, matrix_node, nnz, blk_size, blk_size, values);
  }
  exporter->closeFileNode(matrix_node);

  FileNode vector_node = exporter->createFileNode(root_node, "rhs-0");
  {
    const SimpleCSRVector<Real>& v = rhs.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, vector_node, nrows, blk_size, values);
  }
  exporter->closeFileNode(vector_node);
  _endDump(exporter, root_node);
}

void SystemWriter::dump(BlockMatrix const& A, BlockVector const& rhs, BlockVector const& sol,
                        SolutionInfo const& sol_info)
{
  const SimpleCSRMatrix<Real>& csr = A.impl()->get<BackEnd::tag::simplecsr>();
  const SimpleCSRMatrix<Real>::ProfileType& profile = csr.getProfile();
  int nrows = profile.getNRows();
  int blk_size = A.block().size();
  int nnz = profile.getNnz();
  typedef Exporter::FileNode FileNode;
  Exporter* exporter;
  FileNode root_node;

  _beginDump(exporter, root_node);

  FileNode matrix_node = exporter->createFileNode(root_node, "matrix");
  {
    _writeMatrixInfo(*exporter, matrix_node, nrows, nrows, nnz, blk_size, blk_size);

    const int* cols = profile.cols();
    const int* kcol = profile.kcol();
    _writeCSRProfile(*exporter, matrix_node, nrows, nnz, kcol, cols);

    const double* values = csr.getAddressData();
    _writeMatrixValues(*exporter, matrix_node, nnz, blk_size, blk_size, values);
  }
  exporter->closeFileNode(matrix_node);

  FileNode rhs_node = exporter->createFileNode(root_node, "rhs-0");
  {
    const SimpleCSRVector<Real>& v = rhs.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, rhs_node, nrows, blk_size, values);
  }

  exporter->closeFileNode(rhs_node);

  FileNode sol_node = exporter->createFileNode(root_node, "sol");
  {
    const SimpleCSRVector<Real>& v = sol.impl()->get<BackEnd::tag::simplecsr>();
    const double* values = v.getAddressData();
    _writeVector(*exporter, sol_node, nrows, blk_size, values);
  }
  exporter->closeFileNode(rhs_node);

  _endDump(exporter, root_node);
}

} /* namespace Alien */
