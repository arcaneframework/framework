// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephIni.h                                                       (C) 2010 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_ARCANE_H
#define ALEPH_ARCANE_H
 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Timer.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelSuperMng.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IProcessorAffinityService.h"

#include "arcane/aleph/AlephGlobal.h"
#include "arcane/aleph/AlephTypesSolver.h" 
#include "arcane/aleph/AlephParams.h"

#include "arcane/aleph/Aleph.h"
#include "arcane/aleph/IAlephFactory.h"

#undef ARCANE_HAS_PACKAGE_ITAC
#ifdef ARCANE_HAS_PACKAGE_ITAC
#include </usr/local/intel/itac/7.2.1.008/include/VT.h>
   #define ItacFunction(classname) VT_Function _itac_function_##classname(__func__,#classname)
   #define ItacRegion(region,classname) VT_Region _itac_region_##classname(#region, #classname)
#else
   #define ItacFunction(classname) {}
   #define ItacRegion(region,classname) {}
#endif // ARCANE_HAS_PACKAGE_ITAC

#endif // ALEPH_ARCANE_H
