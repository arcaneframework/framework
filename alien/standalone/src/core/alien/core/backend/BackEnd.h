// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/utils/Precomp.h>
#include <arccore/base/String.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef String BackEndId;

template <typename Tag>
struct AlgebraTraits;

template <typename Tag>
struct LUSendRecvTraits;

namespace BackEnd
{
  namespace tag
  {
  } // namespace tag

  namespace Memory
  {
    typedef enum {
      Host,
      Device,
      Shared
    } eType;
  }

  namespace Exec
  {
    typedef enum {
      Host,
      Device
    } eSpaceType;
  }
} // namespace BackEnd

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
