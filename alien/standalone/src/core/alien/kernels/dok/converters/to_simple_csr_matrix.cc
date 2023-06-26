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

#include "to_simple_csr_matrix.h"

#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/kernels/dok/DoKBackEnd.h>
#include <alien/kernels/dok/DoKMatrixT.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

DoKtoSimpleCSRMatrixConverter::DoKtoSimpleCSRMatrixConverter() {}

DoKtoSimpleCSRMatrixConverter::~DoKtoSimpleCSRMatrixConverter() {}

BackEndId
DoKtoSimpleCSRMatrixConverter::sourceBackend() const
{
  return AlgebraTraits<BackEnd::tag::DoK>::name();
}

BackEndId
DoKtoSimpleCSRMatrixConverter::targetBackend() const
{
  return AlgebraTraits<BackEnd::tag::simplecsr>::name();
}

void DoKtoSimpleCSRMatrixConverter::convert(
const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SrcMatrix& src = cast<SrcMatrix>(sourceImpl, sourceBackend());
  TgtMatrix& tgt = cast<TgtMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting DoKMatrix: " << &src << " to SimpleCSRMatrix: " << &tgt;
  });

  if (sourceImpl->block()) {
    throw FatalErrorException(A_FUNCINFO, "Block matrices are not handled yet");
  }
  else if (sourceImpl->vblock()) {
    throw FatalErrorException(A_FUNCINFO, "Variable block matrices are not handled yet");
  }
  else
    _build(src, tgt);
}

void DoKtoSimpleCSRMatrixConverter::_build(const SrcMatrix& src, TgtMatrix& tgt) const
{
  _buildProfile(src, tgt);
}

void DoKtoSimpleCSRMatrixConverter::_buildProfile(const SrcMatrix& src, TgtMatrix& tgt) const
{
  const ISpace& space = src.rowSpace();
  const MatrixDistribution& dist = src.distribution();
  IMessagePassingMng* parallel_mng = dist.parallelMng();

  if (parallel_mng == nullptr)
    return;

  // if (space != src.colSpace())
  //  throw FatalErrorException("Matrix profiler must be used with square matrix");

  Integer nproc = parallel_mng->commSize();

  // Ugly const_cast but in need for DoKReverseIndexer
  const_cast<SrcMatrix&>(src).compact();

  const Integer local_size = dist.localRowSize();
  const Integer global_size = dist.globalRowSize();
  const Integer local_offset = dist.rowOffset();

  const DoKLocalMatrixT<Real>& dokMatrix = src.data();
  IReverseIndexer* dokMatrixRIndexer = dokMatrix.getReverseIndexer();

  UniqueArray<Integer> offsets(nproc + 1);
  for (int p = 0; p < nproc; p++) {
    offsets[p] = dist.rowOffset(p);
  }
  offsets[nproc] = global_size;

  SimpleCSRInternal::CSRStructInfo& profile = tgt.internal().getCSRProfile();
  profile.init(local_size);

  ArrayView<Integer> row_offsets = profile.getRowOffset();
  row_offsets.fill(0);
  for (Integer i = 0; i < dokMatrixRIndexer->size(); ++i) {
    // FIXME: check if index is correct.
    const Integer localRow = (*dokMatrixRIndexer)[i].value().first + 1 - local_offset;
    ++row_offsets[localRow];
  }
  for (Integer i = 0; i < local_size; ++i)
    row_offsets[i + 1] += row_offsets[i];

  profile.allocate();
  ArrayView<Integer> cols = profile.getCols();

  tgt.allocate();
  ArrayView<Real> values = tgt.internal().getValues();
  values.fill(0);

  for (Integer i = 0; i < dokMatrixRIndexer->size(); ++i) {
    // This is not useful currently. offsetOfIJ == i !
    //    const Integer offsetOfIJ =
    //    dokMatrix.getIndexer()->findOffset((*dokMatrixRIndexer)[i].first,
    //    (*dokMatrixRIndexer)[i].second); cols[offsetOfIJ] =
    //    (*dokMatrixRIndexer)[i].second;
    // FIXME: check if id exists
    cols[i] = (*dokMatrixRIndexer)[i].value().second;
  }

  ConstArrayView<Real> dokValues = dokMatrix.getValues();
  UniqueArray<Real>& csrValues = tgt.internal().getValues();
  for (Integer i = 0; i < dokMatrixRIndexer->size(); ++i) {
    csrValues[i] = dokValues[i];
  }

  if (nproc > 1)
    tgt.parallelStart(offsets, parallel_mng, true);
  else
    tgt.sequentialStart();

  profile.getColOrdering() = SimpleCSRInternal::CSRStructInfo::eFull;
}

REGISTER_MATRIX_CONVERTER(DoKtoSimpleCSRMatrixConverter);

} // namespace Alien
