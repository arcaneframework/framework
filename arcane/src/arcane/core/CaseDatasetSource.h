// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDatasetSource.h                                         (C) 2000-2021 */
/*                                                                           */
/* Source d'un jeu de données d'un cas.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEDATASETSOURCE_H
#define ARCANE_CASEDATASETSOURCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Source d'un jeu de données d'un cas.
 *
 * Il est possible de positionner soit le nom du fichier (setFileName()) ou
 * directement le contenu (setContent()).
 *
 * Si content() est vide et que fileName() est non nul, le jeu de données
 * sera lu par %Arcane au moment du lancement de l'application.
 */
class ARCANE_CORE_EXPORT CaseDatasetSource
{
  class Impl;
 public:
  CaseDatasetSource();
  CaseDatasetSource(const CaseDatasetSource& rhs);
  CaseDatasetSource& operator=(const CaseDatasetSource& rhs);
  ~CaseDatasetSource();
 public:
  //! Positionne le nom du fichier du jeu de données.
  void setFileName(const String& name);
  //! Nom du fichier du jeu de données
  String fileName() const;
  //! Positionne le contenu du jeu de données.
  void setContent(Span<const std::byte> bytes);
  //! Positionne le contenu du jeu de données.
  void setContent(Span<const Byte> bytes);
  //! Contenu du jeu de données.
  ByteConstSpan content() const;
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

