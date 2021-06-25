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

#include "CompositeVectorElement.h"

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/data/IVector.h>
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

  void VectorElement::_assign(IVector* v)
  {
    alien_debug([&] { cout() << "Assign Vector " << this << " in CompositeVector"; });
    m_element.reset(v);
    m_space = m_element->space().clone();
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
