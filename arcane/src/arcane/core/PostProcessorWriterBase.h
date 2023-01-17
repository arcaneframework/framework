// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PostProcessorWriterBase.h                                   (C) 2000-2018 */
/*                                                                           */
/* Classe de base d'un écrivain pour les informations de post-traitement.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_POSTPROCESSORWRITERBASE_H
#define ARCANE_POSTPROCESSORWRITERBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicService.h"
#include "arcane/IPostProcessorWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
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
 public:

  PostProcessorWriterBase(const ServiceBuildInfo& sbi);
  virtual ~PostProcessorWriterBase();

 public:

  virtual void build(){}

 public:

  virtual void setBaseDirectoryName(const String& dirname);
  virtual const String& baseDirectoryName();

  virtual void setBaseFileName(const String& filename);
  virtual const String& baseFileName();

  virtual void setTimes(RealConstArrayView times);
  virtual RealConstArrayView times();

  virtual void setVariables(VariableCollection variables);
  virtual VariableCollection variables();

  virtual void setGroups(ItemGroupCollection groups);
  virtual ItemGroupCollection groups();

 private:

  PostProcessorWriterBasePrivate* m_p; //! Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

