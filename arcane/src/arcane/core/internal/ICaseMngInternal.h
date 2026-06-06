// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMngInternal.h                                          (C) 2000-2025 */
/*                                                                           */
/* Internal Arcane component of ICaseMng.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ICASEMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_ICASEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterListWithCaseOption;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal part of ICaseMng.
 */
class ARCANE_CORE_EXPORT ICaseMngInternal
{
 public:

  virtual ~ICaseMngInternal() = default;

 public:

  /*!
   * \brief Reads an option from the dataset.
   */
  virtual void internalReadOneOption(ICaseOptions* opt, bool is_phase1) = 0;

  /*!
   * \brief Creates a fragment.
   *
   * The returned instance must be destroyed by calling delete.
   * The returned instance becomes the owner of \a document and will be responsible
   * for destroying it.
   */
  virtual ICaseDocumentFragment* createDocumentFragment(IXmlDocumentHolder* document) = 0;

  //! List of parameters that can override the dataset
  virtual const ParameterListWithCaseOption& parameters() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
