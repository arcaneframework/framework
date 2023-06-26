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

/*!
 * \file MatrixConverterRegisterer.h
 * \brief MatrixConverterRegisterer.h
 */
#pragma once

#include <alien/core/backend/IMatrixConverter.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Matrix converter registerer
 *
 * Allows to register a matrix converter to convert a matrix from a format to another
 */
class ALIEN_EXPORT MatrixConverterRegisterer
{
 public:
  //! Type of the matrix converter function
  typedef IMatrixConverter* (*ConverterCreateFunc)();
  //! Type of the backend if
  typedef Alien::BackEndId BackEndId;

 public:
  /*!
   * \brief Creates a matrix converter registerer
   * \param[in] func Matrix converter function
   */
  explicit MatrixConverterRegisterer(ConverterCreateFunc func);

  //! Free resources
  ~MatrixConverterRegisterer() = default;

 public:
  /*!
   * \brief Get the converter from one matrix format to another one
   * \param[in] from Backend id of the source format
   * \param[in] to Backend id of the target format
   * \returns Matrix format converter
   */
  static IMatrixConverter* getConverter(BackEndId from, BackEndId to);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to register a matrix converter
 */
#define REGISTER_MATRIX_CONVERTER(converter) \
  extern "C++" Alien::IMatrixConverter* alienCreateMatrixConverter_##converter() \
  { \
    return new converter(); \
  } \
  Alien::MatrixConverterRegisterer globaliMatrixConverterRegisterer_##converter( \
  alienCreateMatrixConverter_##converter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
