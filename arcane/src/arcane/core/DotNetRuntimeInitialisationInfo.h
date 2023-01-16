﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DotNetRuntimeInitialisationInfo.h                           (C) 2000-2020 */
/*                                                                           */
/* Informations pour l'initialisation du runtime '.Net'.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DOTNETRUNTIMEINITIALISATIONINFO_H
#define ARCANE_UTILS_DOTNETRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/PropertyDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour l'initialisation du runtime '.Net'.
 */
class ARCANE_CORE_EXPORT DotNetRuntimeInitialisationInfo
{
  ARCANE_DECLARE_PROPERTY_CLASS(DotNetRuntimeInitialisationInfo);
 private:
  class Impl;
 public:

  DotNetRuntimeInitialisationInfo();
  DotNetRuntimeInitialisationInfo(const DotNetRuntimeInitialisationInfo& rhs);
  ~DotNetRuntimeInitialisationInfo();
  DotNetRuntimeInitialisationInfo& operator=(const DotNetRuntimeInitialisationInfo& rhs);

 public:

  void setIsUsingDotNetRuntime(bool v);
  bool isUsingDotNetRuntime() const;

  void setMainAssemblyName(StringView name);
  String mainAssemblyName() const;

  void setExecuteClassName(StringView name);
  String executeClassName() const;

  void setExecuteMethodName(StringView name);
  String executeMethodName() const;

  //! Nom du runtime pour le mode embarqué ('mono' ou 'coreclr')
  void setEmbeddedRuntime(StringView name);
  String embeddedRuntime() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

