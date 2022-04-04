﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeGlobal.h                                           (C) 2000-2020 */
/*                                                                           */
/* Définitions globales de la composante 'Serialize' de 'Arccore'.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_SERIALIZEGLOBAL_H
#define ARCCORE_SERIALIZE_SERIALIZEGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_serialize)
#define ARCCORE_SERIALIZE_EXPORT ARCCORE_EXPORT
#define ARCCORE_SERIALIZE_EXTERN_TPL
#else
#define ARCCORE_SERIALIZE_EXPORT ARCCORE_IMPORT
#define ARCCORE_SERIALIZE_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISerializer;
class BasicSerializer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

