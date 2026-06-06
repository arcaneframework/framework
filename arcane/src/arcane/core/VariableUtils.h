// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtils.h                                             (C) 2000-2024 */
/*                                                                           */
/* Various utility functions for variables.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEUTILS_H
#define ARCANE_CORE_VARIABLEUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::VariableUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Prefetches the memory associated with the variable \a var.
 *
 * Prefetches the memory associated with the variable \a onto the accelerator
 * specified by \a queue_or_null if it is not already there.
 *
 * \a var must be a numerical type variable.
 * If \a queue_or_null is null, no operation is performed.
 * The operation is asynchronous.
 */
extern "C++" ARCANE_CORE_EXPORT void prefetchVariableAsync(IVariable* var, const RunQueue* queue_or_null);

/*!
 * \brief Prefetches the memory associated with the variable \a var.
 * \sa void prefetchVariableAsync(IVariable* var, RunQueue* queue_or_null);
 */
extern "C++" ARCANE_CORE_EXPORT void prefetchVariableAsync(VariableRef& var, const RunQueue* queue_or_null);

/*!
 * \brief Indicates that the variable is mostly read-only.
 *
 * This is used only with accelerators and prevents memory transfers
 * between the accelerator and the CPU.
 */
extern "C++" ARCANE_CORE_EXPORT void markVariableAsMostlyReadOnly(IVariable* var);

/*!
 * \brief Indicates that the variable is mostly read-only.
 * \a void markVariableAsMostlyReadOnly(IVariableRef* var);
 */
extern "C++" ARCANE_CORE_EXPORT void markVariableAsMostlyReadOnly(VariableRef& var);

/*!
 * \brief Indicates that the variable is mostly read-only.
 * \a void markVariableAsMostlyReadOnly(IVariableRef* var);
 */
extern "C++" ARCANE_CORE_EXPORT void markVariableAsMostlyReadOnly(::Arcane::Materials::MeshMaterialVariableRef& var);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void experimentalChangeAllocator(::Arcane::Materials::IMeshMaterialVariable* var,
                                                                 eMemoryRessource mem);

extern "C++" ARCANE_CORE_EXPORT void experimentalChangeAllocator(IVariable* var, eMemoryRessource mem);

extern "C++" ARCANE_CORE_EXPORT void experimentalChangeAllocator(VariableRef& var, eMemoryRessource mem);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VariableUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
