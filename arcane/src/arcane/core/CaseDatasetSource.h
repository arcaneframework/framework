// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDatasetSource.h                                         (C) 2000-2025 */
/*                                                                           */
/* Source of a case dataset.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEDATASETSOURCE_H
#define ARCANE_CORE_CASEDATASETSOURCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Source of a case dataset.
 *
 * It is possible to set either the file name (setFileName()) or
 * directly the content (setContent()).
 *
 * If content() is empty and fileName() is not null, the dataset
 * will be read by %Arcane when the application starts.
 */
class ARCANE_CORE_EXPORT CaseDatasetSource
{
  class Impl;
 public:
  CaseDatasetSource();
  CaseDatasetSource(const CaseDatasetSource& rhs);
  CaseDatasetSource& operator=(const CaseDatasetSource& rhs);
  ~CaseDatasetSource();
 public:
  //! Sets the file name of the dataset.
  void setFileName(const String& name);
  //! File name of the dataset
  String fileName() const;
  //! Sets the content of the dataset.
  void setContent(Span<const std::byte> bytes);
  //! Sets the content of the dataset.
  void setContent(Span<const Byte> bytes);
  //! Content of the dataset.
  ByteConstSpan content() const;
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
