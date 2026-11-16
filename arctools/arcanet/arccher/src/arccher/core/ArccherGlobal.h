// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccherGlobal.h                                             (C) 2000-2026 */
/*                                                                           */
/* TODO.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCHER_CORE_ARCCHERGLOBAL_H
#define ARCCHER_CORE_ARCCHERGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * From ArccoreGlobal.h.
 */

#if defined(_WIN32) || defined(__CYGWIN__)
#define ARCCHER_EXPORT __declspec(dllexport)
#define ARCCHER_IMPORT __declspec(dllimport)
#else
#define ARCCHER_EXPORT __attribute__((visibility("default")))
#define ARCCHER_IMPORT __attribute__((visibility("default")))
#endif

#ifndef ARCCHER_EXPORT
#define ARCCHER_EXPORT
#endif

#ifndef ARCCHER_IMPORT
#define ARCCHER_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
