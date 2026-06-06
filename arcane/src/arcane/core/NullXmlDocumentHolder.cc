// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullXmlDocumentHolder.cc                                    (C) 2000-2010 */
/*                                                                           */
/* Manager of a null DOM document.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
