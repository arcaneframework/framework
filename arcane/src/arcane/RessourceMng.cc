// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RessourceMng.cc                                             (C) 2000-2008 */
/*                                                                           */
/* Gestionnaire de ressources.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/IApplication.h"
#include "arcane/IRessourceMng.h"

#include "arcane/DomUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestionnaire de ressources.
 */
class RessourceMng
: public IRessourceMng
{
 public:

  RessourceMng(IApplication* sm);
  virtual ~RessourceMng() {}

  virtual IXmlDocumentHolder* createXmlDocument();

 private:

  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRessourceMng* IRessourceMng::
createDefault(IApplication* sm)
{
  IRessourceMng* m = new RessourceMng(sm);
  return m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RessourceMng::
RessourceMng(IApplication* sm)
: m_application(sm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* RessourceMng::
createXmlDocument()
{
  return domutils::createXmlDocument();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
