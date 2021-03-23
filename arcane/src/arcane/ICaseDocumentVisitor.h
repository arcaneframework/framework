// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseDocumentVisitor.h                                      (C) 2000-2019 */
/*                                                                           */
/* Interface du visiteur pour un jeu de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEDOCUMENTVISITOR_H
#define ARCANE_ICASEDOCUMENTVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseOptionList;
class CaseOptionSimple;
class CaseOptionMultiSimple;
class CaseOptionExtended;
class CaseOptionMultiExtended;
class CaseOptionEnum;
class CaseOptionMultiEnum;
class CaseOptionServiceImpl;
class CaseOptionMultiServiceImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du visiteur pour une option du jeu de données.
 */
class ARCANE_CORE_EXPORT ICaseDocumentVisitor
{
 public:
  
  virtual ~ICaseDocumentVisitor(){}

 public:

  virtual void beginVisit(const ICaseOptions* opt) =0;
  virtual void endVisit(const ICaseOptions* opt) =0;
  virtual void beginVisit(const CaseOptionServiceImpl* opt) =0;
  virtual void endVisit(const CaseOptionServiceImpl* opt) =0;
  virtual void beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index) =0;
  virtual void endVisit(const CaseOptionMultiServiceImpl* opt,Integer index) =0;
  virtual void applyVisitor(const CaseOptionSimple* opt) =0;
  virtual void applyVisitor(const CaseOptionMultiSimple* opt) =0;
  virtual void applyVisitor(const CaseOptionExtended* opt) =0;
  virtual void applyVisitor(const CaseOptionMultiExtended* opt) =0;
  virtual void applyVisitor(const CaseOptionEnum* opt) =0;
  virtual void applyVisitor(const CaseOptionMultiEnum* opt) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

