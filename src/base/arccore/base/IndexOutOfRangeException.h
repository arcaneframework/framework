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
/* IndexOutOfRangeException.h                                  (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un indice de tableau est invalide.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INDEXOUTOFRANGEEXCEPTION_H
#define ARCCORE_BASE_INDEXOUTOFRANGEEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Exception lorsqu'une erreur fatale est survenue.
 */
class ARCCORE_BASE_EXPORT IndexOutOfRangeException
: public Exception
{
 public:
	
  IndexOutOfRangeException(const TraceInfo& where,const String& message,
                           Int64 index,Int64 min_value,Int64 max_value);
  IndexOutOfRangeException(const IndexOutOfRangeException& ex);
  ~IndexOutOfRangeException() ARCCORE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;
  
  Int64 index() const { return m_index; }
  Int64 minValue() const { return m_min_value; }
  Int64 maxValue() const { return m_max_value; }

 private:

  Int64 m_index;
  Int64 m_min_value;
  Int64 m_max_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

