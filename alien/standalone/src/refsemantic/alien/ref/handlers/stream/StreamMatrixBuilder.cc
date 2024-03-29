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

#include <alien/ref/handlers/stream/StreamMatrixBuilder.h>

#undef ALIEN_IFPEN_STREAM_STREAMMATRIXBUILDER_CC

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ALIEN_REFSEMANTIC_EXPORT StreamMatrixBuilderT<double>;

#ifdef WIN32
// These exports do not work on Win32 (july 2023). A fix is needed
#ifdef FIXED_BUGGY_EXPORT
template class ALIEN_IFPEN_EXPORT StreamMatrixBuilderT<double>::BaseInserter;
template class ALIEN_IFPEN_EXPORT StreamMatrixBuilderT<double>::Profiler;
template class ALIEN_IFPEN_EXPORT StreamMatrixBuilderT<double>::Filler;
template class ALIEN_IFPEN_EXPORT StreamMatrixBuilderT<double>::Inserter;

template ALIEN_IFPEN_EXPORT StreamMatrixBuilderT<double>::Filler&
StreamMatrixBuilderT<double>::Filler::operator++();
template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Filler::start();
template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Filler::addBlockData(
ConstArrayView<double> values);
template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Filler::addData(
double values);
template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Filler::addData(
ConstArrayView<double> values, double factor);

template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Profiler::addMatrixEntry(
Integer row_index, Integer col_index);
template ALIEN_IFPEN_EXPORT void StreamMatrixBuilderT<double>::Profiler::addMatrixEntries(
const Integer row_index, ConstArrayView<Integer> col_index);
template ALIEN_IFPEN_EXPORT Integer
StreamMatrixBuilderT<double>::BaseInserter::getId() const;
#endif

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
