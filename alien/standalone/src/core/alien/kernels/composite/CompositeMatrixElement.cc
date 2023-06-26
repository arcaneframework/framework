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

#include "CompositeMatrixElement.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/data/IMatrix.h>
#include <alien/data/ISpace.h>
#include <alien/utils/Trace.h>
#include <alien/utils/time_stamp/TimestampObserver.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  void MatrixElement::_assign(IMatrix* v)
  {
    alien_debug([&] { cout() << "Assign Matrix " << this << " in CompositeMatrix"; });

    m_element.reset(v);
    m_row_space = m_element->rowSpace().clone();
    m_col_space = m_element->colSpace().clone();
    m_element->impl()->addObserver(std::make_shared<TimestampObserver>(m_timestamp));
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
