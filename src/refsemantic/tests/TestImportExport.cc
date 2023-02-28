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
#include <fstream>

#include <gtest/gtest.h>

#include <alien/ref/AlienImportExport.h>
#include <alien/ref/AlienRefSemantic.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <Environment.h>

using namespace Arccore;

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
    Alien::SystemWriter writer("SystemAb.txt");
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
    Alien::SystemWriter writer("SystemAbx.txt");
    Alien::SolutionInfo sol_info(Alien::SolutionInfo::N2_RELATIVE2RHS_RES, 1e-7, "fake");
    writer.dump(A, b, x, sol_info);
  }
}

TEST(TestImportExport, ImportMatrixMarketMatrix)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    const std::string mat =
    "%%MatrixMarket matrix coordinate real general\n"
    "%-------------------------------------------------------------------------------\n"
    "% UF Sparse Matrix Collection, Tim Davis\n"
    "% http://www.cise.ufl.edu/research/sparse/matrices/vanHeukelum/cage4\n"
    "% name: vanHeukelum/cage4\n"
    "% [DNA electrophoresis, 4 monomers in polymer. A. van Heukelum, Utrecht U.]\n"
    "% id: 905\n"
    "% date: 2003\n"
    "% author: A. van Heukelum\n"
    "% ed: T. Davis\n"
    "% fields: title A name id date author ed kind\n"
    "% kind: directed weighted graph\n"
    "%-------------------------------------------------------------------------------\n"
    "9 9 49\n"
    "1 1 .75\n"
    "2 1 .075027667114587\n"
    "4 1 .0916389995520797\n"
    "5 1 .0375138335572935\n"
    "8 1 .0458194997760398\n"
    "1 2 .137458499328119\n"
    "2 2 .687569167786467\n"
    "3 2 .0916389995520797\n"
    "5 2 .0375138335572935\n"
    "6 2 .0458194997760398\n"
    "2 3 .112541500671881\n"
    "3 3 .666666666666667\n"
    "4 3 .13745849932812\n"
    "6 3 .0458194997760398\n"
    "7 3 .0375138335572935\n"
    "1 4 .112541500671881\n"
    "3 4 .075027667114587\n"
    "4 4 .729097498880199\n"
    "7 4 .0375138335572935\n"
    "8 4 .0458194997760398\n"
    "1 5 .137458499328119\n"
    "2 5 .075027667114587\n"
    "5 5 .537513833557293\n"
    "6 5 .075027667114587\n"
    "7 5 .0916389995520797\n"
    "9 5 .0833333333333333\n"
    "2 6 .112541500671881\n"
    "3 6 .0916389995520797\n"
    "5 6 .13745849932812\n"
    "6 6 .445874834005214\n"
    "8 6 .13745849932812\n"
    "9 6 .075027667114587\n"
    "3 7 .075027667114587\n"
    "4 7 .13745849932812\n"
    "5 7 .112541500671881\n"
    "7 7 .470791832661453\n"
    "8 7 .112541500671881\n"
    "9 7 .0916389995520797\n"
    "1 8 .112541500671881\n"
    "4 8 .0916389995520797\n"
    "6 8 .075027667114587\n"
    "7 8 .0916389995520797\n"
    "8 8 .54581949977604\n"
    "9 8 .0833333333333333\n"
    "5 9 .25\n"
    "6 9 .150055334229174\n"
    "7 9 .183277999104159\n"
    "8 9 .25\n"
    "9 9 .166666666666667\n";

    {
      std::fstream matrix_file_stream("cage4.mtx", std::ios_base::out);
      matrix_file_stream << mat;
    }

    Alien::Matrix A;

    Alien::MatrixMarketSystemReader reader("cage4.mtx");

    reader.read(A);

    ASSERT_EQ(9, A.rowSpace().size());
    ASSERT_EQ(9, A.colSpace().size());

    const auto& A_csr = A.impl()->get<Alien::BackEnd::tag::simplecsr>();

    ASSERT_EQ(49, A_csr.getProfile().getNElems());

    system("rm cage4.mtx");
  }
}

TEST(TestImportExport, ImportMatrixMarketRhs)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    const std::string rhs =
    "%%MatrixMarket matrix array real general\n"
    "%-------------------------------------------------------------------------------\n"
    "% Fake rhs for test\n"
    "%-------------------------------------------------------------------------------\n"
    "9 1\n"
    ".75\n"
    ".075027667114587\n"
    ".0916389995520797\n"
    ".0375138335572935\n"
    ".0458194997760398\n"
    ".137458499328119\n"
    ".687569167786467\n"
    ".0916389995520797\n"
    ".0375138335572935\n";

    {
      std::fstream rhs_file_stream("vec_b.mtx", std::ios_base::out);
      rhs_file_stream << rhs;
    }

    Alien::Vector vec;

    Alien::MatrixMarketSystemReader reader("vec_b.mtx");

    reader.read(vec);

    ASSERT_EQ(9, vec.space().size());

    system("rm vec_b.mtx");
  }
}

#ifdef ALIEN_USE_LIBARCHIVE
TEST(TestImportExport, ImportSuiteSparseArchive)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    const std::string mat =
    "%%MatrixMarket matrix coordinate real general\n"
    "%-------------------------------------------------------------------------------\n"
    "% UF Sparse Matrix Collection, Tim Davis\n"
    "% http://www.cise.ufl.edu/research/sparse/matrices/Grund/b1_ss\n"
    "% name: Grund/b1_ss\n"
    "% [Unsymmetric Matrix b1_ss, F. Grund, Dec 1994.]\n"
    "% id: 449\n"
    "% date: 1997\n"
    "% author: F. Grund\n"
    "% ed: F. Grund\n"
    "% fields: title A b name id date author ed kind\n"
    "% kind: chemical process simulation problem\n"
    "%-------------------------------------------------------------------------------\n"
    "7 7 15\n"
    "5 1 -.03599942\n"
    "6 1 -.0176371\n"
    "7 1 -.007721779\n"
    "1 2 1\n"
    "2 2 -1\n"
    "1 3 1\n"
    "3 3 -1\n"
    "1 4 1\n"
    "4 4 -1\n"
    "2 5 .45\n"
    "5 5 1\n"
    "3 6 .1\n"
    "6 6 1\n"
    "4 7 .45\n"
    "7 7 1\n";

    const std::string rhs =
    "%%MatrixMarket matrix array real general\n"
    "%-------------------------------------------------------------------------------\n"
    "% UF Sparse Matrix Collection, Tim Davis\n"
    "% http://www.cise.ufl.edu/research/sparse/matrices/Grund/b1_ss\n"
    "% name: Grund/b1_ss : b matrix\n"
    "%-------------------------------------------------------------------------------\n"
    "7 1\n"
    "-.0001\n"
    ".1167\n"
    "-.2333\n"
    ".1167\n"
    "-.4993128\n"
    ".3435885\n"
    ".7467878\n";

    system("rm -rf b1_ss");
    system("mkdir b1_ss");
    system("pwd");
    {
      std::fstream mat_file_stream("b1_ss/b1_ss.mtx", std::ios_base::out);
      mat_file_stream << mat;
      std::fstream rhs_file_stream("b1_ss/b1_ss_b.mtx", std::ios_base::out);
      rhs_file_stream << rhs;
    }

    system("tar -zcf b1_ss.tar.gz b1_ss");
    system("rm -r b1_ss");

    Alien::SuiteSparseArchiveSystemReader archive_system_reader("b1_ss.tar.gz");

    Alien::Matrix A;
    Alien::Vector vec;

    archive_system_reader.read(A);
    archive_system_reader.read(vec);

    ASSERT_EQ(7, A.rowSpace().size());
    ASSERT_EQ(7, A.colSpace().size());

    ASSERT_EQ(7, vec.space().size());

    system("rm b1_ss.tar.gz");
  }
}
#endif