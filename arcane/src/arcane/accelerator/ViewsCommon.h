// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ViewsCommon.h                                               (C) 2000-2024 */
/*                                                                           */
/* Types de base pour la gestion des vues pour les accélérateurs.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_VIEWSCOMMON_H
#define ARCANE_ACCELERATOR_VIEWSCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/core/DataView.h"
#include "arcane/accelerator/core/ViewBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ViewsCommon.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> using DataViewSetter = Arcane::DataViewSetter<DataType>;
template <typename DataType> using DataViewGetterSetter = Arcane::DataViewGetterSetter<DataType>;
template <typename DataType> using DataViewGetter = Arcane::DataViewGetter<DataType>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
