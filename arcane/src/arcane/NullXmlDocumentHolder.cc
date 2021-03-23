// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullXmlDocumentHolder.cc                                    (C) 2000-2010 */
/*                                                                           */
/* Gestionnaire d'un document DOM null.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"

#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullXmlDocumentHolder
: public IXmlDocumentHolder
{
 public:
  virtual XmlNode documentNode() { return XmlNode(); }
  virtual IXmlDocumentHolder* clone() { return new NullXmlDocumentHolder(); }
  virtual void save(ByteArray& bytes) { bytes.clear(); }
  virtual String save() { return String(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IXmlDocumentHolder::
createNull()
{
  return new NullXmlDocumentHolder();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

