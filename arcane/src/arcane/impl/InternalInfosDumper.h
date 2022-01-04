// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InternalInfosDumper.h                                       (C) 2000-2019 */
/*                                                                           */
/* Sorties des informations internes de Arcane.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNALINFOSDUMPER_H
#define ARCANE_IMPL_INTERNALINFOSDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;
class ICodeService;
class ISubDomain;
class JSONWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sorties des informations internes de Arcane.
 */
class ARCANE_IMPL_EXPORT InternalInfosDumper
{
 public:

  InternalInfosDumper(IApplication* application);

 public:

  void dumpInternalInfos();
  void dumpInternalAllInfos();
  void dumpArcaneDatabase();

 private:

  IApplication* m_application;

 private:

  Ref<ICodeService> _getDefaultService();
  void _dumpSubDomainInternalInfos(ISubDomain* sd,JSONWriter& json_writer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
