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
#include <gtest/gtest.h>

#include <alien/import_export/MatrixMarketSystemReader.h>
#ifdef ALIEN_USE_LIBARCHIVE
#include <alien/import_export/SuiteSparseArchiveSystemReader.h>
#endif

#include <alien/move/data/MatrixData.h>
#include <alien/move/data/VectorData.h>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <Environment.h>
#include <CreateLinearSystemFiles.h>

using namespace Arccore;

TEST(TestImportExport, ImportMatrixMarketMatrix)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    createMMMatrixFile("cage4.mtx");

    Alien::Move::MatrixData A;

    Alien::MatrixMarketSystemReader reader("cage4.mtx");

    reader.readMatrix(A);

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
    createMMRhsFile("vec_b.mtx");

    Alien::Move::VectorData vec;

    Alien::MatrixMarketSystemReader reader("vec_b.mtx");

    reader.readVector(vec);

    ASSERT_EQ(9, vec.space().size());

    system("rm vec_b.mtx");
  }
}

#ifdef ALIEN_USE_LIBARCHIVE
TEST(TestImportExport, ImportSuiteSparseArchive)
{
  if (AlienTest::Environment::parallelMng()->commRank() == 0) {
    createSSArchive("b1_ss");

    Alien::SuiteSparseArchiveSystemReader archive_system_reader("b1_ss.tar.gz");

    Alien::Move::MatrixData A;
    Alien::Move::VectorData vec;

    archive_system_reader.readMatrix(A);
    archive_system_reader.readVector(vec);

    ASSERT_EQ(7, A.rowSpace().size());
    ASSERT_EQ(7, A.colSpace().size());

    ASSERT_EQ(7, vec.space().size());

    system("rm b1_ss.tar.gz");
  }
}
#endif