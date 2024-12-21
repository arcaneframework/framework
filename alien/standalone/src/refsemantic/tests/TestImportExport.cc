// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*
 * Copyright 2024 IFPEN-CEA
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
#include <gtest/gtest.h>

#include <arccore/base/PlatformUtils.h>

#include <alien/import_export/MatrixMarketSystemReader.h>
#ifdef ALIEN_USE_LIBARCHIVE
#include <alien/import_export/SuiteSparseArchiveSystemReader.h>
#endif

#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <Environment.h>
#include <CreateLinearSystemFiles.h>

using namespace Arccore;

namespace
{

std::string _getProcessUniqueFilename(const char* base_name,const char* extension = "")
{
  // Use a unique filename because tests are running in parallel and we do not want
  // to overwrite files from other tests
  Int32 pid = Platform::getProcessId();
  String file_name = String::format("{0}.{1}{2}",base_name,pid,extension);
  return file_name.localstr();
}

}
TEST(TestImportExport, ImportExportMatrix)
{
  Alien::Space row_space(10, "RowSpace");
  Alien::MatrixDistribution mdist(
  row_space, row_space, AlienTest::Environment::parallelMng());

  Alien::Matrix A(mdist);
  auto local_size = mdist.localRowSize();
  auto global_size = mdist.globalColSize();
  auto offset = mdist.rowOffset();
  {
    Alien::MatrixProfiler profiler(A);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      profiler.addMatrixEntry(row, row);
      if (row + 1 < global_size)
        profiler.addMatrixEntry(row, row + 1);
      if (row - 1 >= 0)
        profiler.addMatrixEntry(row, row - 1);
    }
  }
  {
    Alien::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }
  {
    Alien::SystemWriter writer("Matrix.txt");
    writer.dump(A);
  }
  if (AlienTest::Environment::parallelMng()->commSize() == 1) {
    Alien::Matrix B(mdist);
    {
      Alien::SystemReader reader("Matrix.txt");
      reader.read(B);
      Alien::ProfiledMatrixBuilder a_view(A, Alien::ProfiledMatrixOptions::eKeepValues);
      Alien::ProfiledMatrixBuilder b_view(B, Alien::ProfiledMatrixOptions::eKeepValues);

      for (Integer i = 0; i < local_size; ++i) {
        Integer row = offset + i;
        ASSERT_EQ(a_view(row, row)(), b_view(row, row)());
        if (row + 1 < global_size) {
          ASSERT_EQ(a_view(row, row + 1)(), b_view(row, row + 1)());
        }
        if (row - 1 >= 0) {
          ASSERT_EQ(a_view(row, row - 1)(), b_view(row, row - 1)());
        }
      }
    }
  }
}

TEST(TestImportExport, ExportSystem)
{
  Alien::Space row_space(10, "RowSpace");
  Alien::MatrixDistribution mdist(
  row_space, row_space, AlienTest::Environment::parallelMng());

  Alien::Matrix A(mdist);
  auto local_size = mdist.localRowSize();
  auto global_size = mdist.globalColSize();
  auto offset = mdist.rowOffset();
  {
    Alien::MatrixProfiler profiler(A);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      profiler.addMatrixEntry(row, row);
      if (row + 1 < global_size)
        profiler.addMatrixEntry(row, row + 1);
      if (row - 1 >= 0)
        profiler.addMatrixEntry(row, row - 1);
    }
  }
  {
    Alien::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }

  Alien::VectorDistribution vdist(row_space, AlienTest::Environment::parallelMng());

  Alien::Vector b(vdist);
  {
    Alien::LocalVectorWriter writer(b);
    for (Integer i = 0; i < vdist.localSize(); ++i) {
      writer[i] = i;
    }
  }
  {
    Alien::SystemWriter writer(_getProcessUniqueFilename("SystemAb",".txt"));
    writer.dump(A, b);
  }

  Alien::Vector x(vdist);
  {
    Alien::LocalVectorWriter writer(x);
    for (Integer i = 0; i < vdist.localSize(); ++i) {
      writer[i] = 1;
    }
  }
  {
    Alien::SystemWriter writer(_getProcessUniqueFilename("SystemAbx",".txt"));
    Alien::SolutionInfo sol_info(Alien::SolutionInfo::N2_RELATIVE2RHS_RES, 1e-7, "fake");
    writer.dump(A, b, x, sol_info);
  }
}

TEST(TestImportExport, ImportMatrixMarketMatrix)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    std::string file_name = _getProcessUniqueFilename("cage4",".mtx");
    createMMMatrixFile(file_name);

    Alien::Matrix A;

    Alien::MatrixMarketSystemReader reader(file_name);

    reader.readMatrix(A);

    ASSERT_EQ(9, A.rowSpace().size());
    ASSERT_EQ(9, A.colSpace().size());

    const auto& A_csr = A.impl()->get<Alien::BackEnd::tag::simplecsr>();

    ASSERT_EQ(49, A_csr.getProfile().getNElems());

    Platform::removeFile(file_name);
  }
}

TEST(TestImportExport, ImportMatrixMarketRhs)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    std::string file_name = _getProcessUniqueFilename("vec_b",".mtx");
    createMMRhsFile(file_name);

    Alien::Vector vec;

    Alien::MatrixMarketSystemReader reader(file_name);

    reader.readVector(vec);

    ASSERT_EQ(9, vec.space().size());

    Platform::removeFile(file_name);
  }
}

#ifdef ALIEN_USE_LIBARCHIVE
TEST(TestImportExport, ImportSuiteSparseArchive)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {

    std::string file_name = _getProcessUniqueFilename("b1_ss");
    createSSArchive(file_name);
    std::string tar_file_name = file_name + ".tar.gz";

    Alien::SuiteSparseArchiveSystemReader archive_system_reader(tar_file_name);

    Alien::Matrix A;
    Alien::Vector vec;

    archive_system_reader.readMatrix(A);
    archive_system_reader.readVector(vec);

    ASSERT_EQ(7, A.rowSpace().size());
    ASSERT_EQ(7, A.colSpace().size());

    ASSERT_EQ(7, vec.space().size());
    Platform::removeFile(tar_file_name);
  }
}
#endif
