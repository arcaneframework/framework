// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractCaseOptionVisitor.h                                 (C) 2000-2025 */
/*                                                                           */
/* Visiteur abstrait pour une option du jeu de données.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTCASEDOCUMENTVISITOR_H
#define ARCANE_CORE_ABSTRACTCASEDOCUMENTVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ICaseDocumentVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée scalaire.
 *
 * Ce visiteur lève une exception pour chaque fonction applyVisitor()
 * non réimplémentée.
 */
class ARCANE_CORE_EXPORT AbstractCaseDocumentVisitor
: public ICaseDocumentVisitor
{
 public:

  void beginVisit(const ICaseOptions* opt) override;
  void endVisit(const ICaseOptions* opt) override;
  void beginVisit(const CaseOptionServiceImpl* opt) override;
  void endVisit(const CaseOptionServiceImpl* opt) override;
  void beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override;
  void endVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override;
  void applyVisitor(const CaseOptionSimple* opt) override;
  void applyVisitor(const CaseOptionMultiSimple* opt) override;
  void applyVisitor(const CaseOptionExtended* opt) override;
  void applyVisitor(const CaseOptionMultiExtended* opt) override;
  void applyVisitor(const CaseOptionEnum* opt) override;
  void applyVisitor(const CaseOptionMultiEnum* opt) override;

 protected:

  void _throwException();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

