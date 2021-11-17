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
 * \file VectorConverterRegisterer.h
 * \brief VectorConverterRegisterer.h
 */
#pragma once

#include <alien/core/backend/IVectorConverter.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Vector converter registerer
 *
 * Allows to register a vector converter to convert a vector from a format to another
 */
class ALIEN_EXPORT VectorConverterRegisterer
{
 public:
  //! Type of the vector converter function
  typedef IVectorConverter* (*ConverterCreateFunc)();

 public:
  /*!
   * \brief Creates a vector converter registerer
   * \param[in] func vector converter function
   */
  explicit VectorConverterRegisterer(ConverterCreateFunc func);

  //! Free resources
  ~VectorConverterRegisterer() = default;

 public:
  /*!
   * \brief Get the converter from one vector format to another one
   * \param[in] from Backend id of the source format
   * \param[in] to Backend id of the target format
   * \returns vector format converter
   */
  static IVectorConverter* getConverter(BackEndId from, BackEndId to);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to register a vector converter
 */
#define REGISTER_VECTOR_CONVERTER(converter) \
  extern "C++" Alien::IVectorConverter* alienCreateVectorConverter_##converter() \
  { \
    return new converter(); \
  } \
  Alien::VectorConverterRegisterer globaliVectorConverterRegisterer_##converter( \
  alienCreateVectorConverter_##converter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
