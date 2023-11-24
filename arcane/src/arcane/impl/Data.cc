// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2021 */
/*                                                                           */
/* Classes de base d'une donnée.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

extern "C++" void
registerStringScalarDataFactory(IDataFactoryMng* dfm);
extern "C++" void
registerStringArrayDataFactory(IDataFactoryMng* dfm);
extern "C++" void
registerScalarDataFactory(IDataFactoryMng* dfm);
extern "C++" void
registerArrayDataFactory(IDataFactoryMng* dfm);
extern "C++" void
registerArray2DataFactory(IDataFactoryMng* dfm);
extern "C++" void
registerNumArrayDataFactory(IDataFactoryMng* dfm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
arcaneRegisterSimpleData(IDataFactoryMng* dfm)
{
  // Enregistre les types de donnée standard.
  registerStringScalarDataFactory(dfm);
  registerStringArrayDataFactory(dfm);
  registerScalarDataFactory(dfm);
  registerArrayDataFactory(dfm);
  registerArray2DataFactory(dfm);
#if defined(ARCANE_HAS_ACCELERATOR_API)
  registerNumArrayDataFactory(dfm);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
