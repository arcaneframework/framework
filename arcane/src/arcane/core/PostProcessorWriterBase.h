// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.h                                   (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un écrivain pour les informations de post-traitement.    */
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
 * \ingroup StandardService
 * \brief Classe de base d'un écrivain pour les informations de post-traitement.
 */
class ARCANE_CORE_EXPORT PostProcessorWriterBase
: public BasicService
, public IPostProcessorWriter
{
 public:

  explicit PostProcessorWriterBase(const ServiceBuildInfo& sbi);
  ~PostProcessorWriterBase() override;

 public:

  void build() override {}

 public:

  void setBaseDirectoryName(const String& dirname) override;
  const String& baseDirectoryName() override;

  void setBaseFileName(const String& filename) override;
  const String& baseFileName() override;

  void setTimes(RealConstArrayView times) override;
  RealConstArrayView times() override;

  void setVariables(VariableCollection variables) override;
  VariableCollection variables() override;

  void setGroups(ItemGroupCollection groups) override;
  ItemGroupCollection groups() override;

 private:

  PostProcessorWriterBasePrivate* m_p = nullptr; //! Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

