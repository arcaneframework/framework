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

#include <arccore/base/ArccoreGlobal.h>
#include <alien/import_export/Reader.h>

#include <string>
#include <archive.h>
#include <archive_entry.h>

namespace Alien
{
class Matrix;
class Vector;

class ALIEN_EXPORT SuiteSparseArchiveSystemReader
{
 public:
  SuiteSparseArchiveSystemReader() = delete;
  SuiteSparseArchiveSystemReader(SuiteSparseArchiveSystemReader const&) = delete;
  SuiteSparseArchiveSystemReader& operator=(SuiteSparseArchiveSystemReader const&) = delete;

  explicit SuiteSparseArchiveSystemReader(std::string const& filename);
  ~SuiteSparseArchiveSystemReader();

  template <typename MatrixT>
  void readMatrix(MatrixT& A);

  template <typename VectorT>
  void readVector(VectorT& rhs);

 private:
  std::string m_filename;
  archive* m_archive = nullptr;
};

template <typename MatrixT>
void SuiteSparseArchiveSystemReader::readMatrix(MatrixT& A)
{
  m_archive = archive_read_new();
  archive_read_support_filter_gzip(m_archive);
  archive_read_support_format_tar(m_archive);

  auto r = archive_read_open_filename(m_archive, m_filename.c_str(), 10240);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Open archive " + m_filename);
  }

  // look for matrix in archive
  bool matrix_found = false;
  archive_entry* entry = nullptr;
  auto pos = m_filename.find_last_of('/');
  std::string matrix_name(m_filename, pos + 1, m_filename.size() - pos - 1 - 7); // .tar.gz

  while (archive_read_next_header(m_archive, &entry) == ARCHIVE_OK) {
    //std::cout << "entry name :" << archive_entry_pathname(entry) << "\n";
    if (archive_entry_pathname(entry) == std::string(matrix_name + "/" + matrix_name + ".mtx")) {
      matrix_found = true;
      break;
    }
    archive_read_data_skip(m_archive);
  }

  if (!matrix_found) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Matrix not found in " + m_filename);
  }

  LibArchiveReader reader(m_archive);
  loadMMMatrixFromReader<MatrixT, LibArchiveReader>(A, reader);

  r = archive_read_close(m_archive);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Close archive " + m_filename);
  }

  r = archive_free(m_archive);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Free archive");
  }
}

template <typename VectorT>
void SuiteSparseArchiveSystemReader::readVector(VectorT& rhs)
{
  m_archive = archive_read_new();
  archive_read_support_filter_gzip(m_archive);
  archive_read_support_format_tar(m_archive);

  auto r = archive_read_open_filename(m_archive, m_filename.c_str(), 10240);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Open archive " + m_filename);
  }

  bool vector_found = false;
  archive_entry* entry = nullptr;
  auto pos = m_filename.find_last_of('/');
  std::string matrix_name(m_filename, pos + 1, m_filename.size() - pos - 1 - 7); // .tar.gz

  while (archive_read_next_header(m_archive, &entry) == ARCHIVE_OK) {
    //std::cout << "entry name :" << archive_entry_pathname(entry) << "\n";
    if (archive_entry_pathname(entry) == std::string(matrix_name + "/" + matrix_name + "_b.mtx")) {
      vector_found = true;
      break;
    }
    archive_read_data_skip(m_archive);
  }

  if (vector_found) // vector is not always present
  {
    LibArchiveReader reader(m_archive);
    loadMMRhsFromReader<VectorT, LibArchiveReader>(rhs, reader);
  }

  r = archive_read_close(m_archive);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Close archive " + m_filename);
  }

  r = archive_free(m_archive);
  if (r != ARCHIVE_OK) {
    throw FatalErrorException(__PRETTY_FUNCTION__, "Free archive");
  }
}

} // namespace Alien
