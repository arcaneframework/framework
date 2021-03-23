// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractCaseOptionVisitor.cc                                (C) 2000-2017 */
/*                                                                           */
/* Visiteur abstrait pour une option du jeu de données.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/AbstractCaseDocumentVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
_throwException()
{
  String s = String::format("visitor not implemented for this case option");
  throw NotImplementedException(A_FUNCINFO,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
beginVisit(const ICaseOptions* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
endVisit(const ICaseOptions* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionSimple* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionMultiSimple* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionExtended* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionMultiExtended* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionEnum* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
applyVisitor(const CaseOptionMultiEnum* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
beginVisit(const CaseOptionServiceImpl* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
endVisit(const CaseOptionServiceImpl* opt)
{
  ARCANE_UNUSED(opt);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index)
{
  ARCANE_UNUSED(opt);
  ARCANE_UNUSED(index);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractCaseDocumentVisitor::
endVisit(const CaseOptionMultiServiceImpl* opt,Integer index)
{
  ARCANE_UNUSED(opt);
  ARCANE_UNUSED(index);
  _throwException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
