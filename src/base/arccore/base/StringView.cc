// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringView.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Vue sur une chaîne de caractères UTF-8.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringView.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringView::
writeBytes(std::ostream& o) const
{
  o.write((const char*)m_v.data(),m_v.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const StringView& str)
{
  str.writeBytes(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
operator==(StringView a,StringView b)
{
  bool is_equal = (a.toStdStringView()==b.toStdStringView());
  //std::cout << "COMPARE: a=" << a.length() << " '" << a << "'"
  //          << " b=" << b.length() << " '" << b << "' v=" << is_equal << '\n';
  return is_equal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool operator==(const char* a,StringView b)
{
  return operator==(StringView(a),b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool operator==(StringView a,const char* b)
{
  return operator==(a,StringView(b));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool operator<(StringView a,StringView b)
{
  return a.toStdStringView()<b.toStdStringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
