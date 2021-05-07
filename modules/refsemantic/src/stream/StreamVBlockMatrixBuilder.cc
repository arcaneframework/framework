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

#include <alien/ref/handlers/stream/StreamVBlockMatrixBuilder.h>

#include <alien/ref/AlienRefSemanticPrecomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template class ALIEN_REFSEMANTIC_EXPORT StreamVBlockMatrixBuilderT<double>;

#ifdef WIN32
template class ALIEN_IFPEN_EXPORT StreamVBlockMatrixBuilderT<double>::BaseInserter;
template class ALIEN_IFPEN_EXPORT StreamVBlockMatrixBuilderT<double>::Profiler;
template class ALIEN_IFPEN_EXPORT StreamVBlockMatrixBuilderT<double>::Filler;
template class ALIEN_IFPEN_EXPORT StreamVBlockMatrixBuilderT<double>::Inserter;

template ALIEN_IFPEN_EXPORT StreamVBlockMatrixBuilderT<double>::Filler&
StreamVBlockMatrixBuilderT<double>::Filler::operator++();
template ALIEN_IFPEN_EXPORT void StreamVBlockMatrixBuilderT<double>::Filler::addBlockData(
ConstArray2View<double> values);
template ALIEN_IFPEN_EXPORT void StreamVBlockMatrixBuilderT<double>::Filler::start();
template ALIEN_IFPEN_EXPORT void
StreamVBlockMatrixBuilderT<double>::Profiler::addMatrixEntry(
Integer row_index, Integer col_index);

#endif

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
