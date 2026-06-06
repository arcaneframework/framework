// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.h                                   (C) 2000-2026 */
/*                                                                           */
/* Base class for a writer for post-processing information.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_POSTPROCESSORWRITERBASE_H
#define ARCANE_CORE_POSTPROCESSORWRITERBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicService.h"
#include "arcane/core/IPostProcessorWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PostProcessorWriterBasePrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for a writer for post-processing information.
 */
class ARCANE_CORE_EXPORT PostProcessorWriterCommonBase
: public IPostProcessorWriter
{
 public:

  PostProcessorWriterCommonBase();
  ~PostProcessorWriterCommonBase() override;

 public:

  void setBaseDirectoryName(const String& dirname) override;
  const String& baseDirectoryName() override;

  void setBaseFileName(const String& filename) override;
  const String& baseFileName() override;

  void setTimes(ConstArrayView<Real> times) override;
  ConstArrayView<Real> times() override;

  void setVariables(VariableCollection variables) override;
  VariableCollection variables() override;

  void setGroups(ItemGroupCollection groups) override;
  ItemGroupCollection groups() override;

 private:

  PostProcessorWriterBasePrivate* m_p = nullptr; //! Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Base class for a writer service for post-processing information.
 */
class ARCANE_CORE_EXPORT PostProcessorWriterBase
: public BasicService
, public PostProcessorWriterCommonBase
{
 public:

  explicit PostProcessorWriterBase(const ServiceBuildInfo& sbi);

 public:

  void build() override {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
