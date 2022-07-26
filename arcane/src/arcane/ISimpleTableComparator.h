// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableOutput.hh                                       (C) 2000-2022 */
/*                                                                           */
/* Interface pour simples services de comparaison de tableaux générés par    */
/* ISimpleTableOutput.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLECOMPARATOR_H
#define ARCANE_ISIMPLETABLECOMPARATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ItemTypes.h>
#include <arcane/ISimpleTableOutput.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Interface représentant un comparateur de tableau simple.
 * @warning Interface non définitive !
 */
class ARCANE_CORE_EXPORT ISimpleTableComparator
{
public:
  virtual ~ISimpleTableComparator() = default;

public:
  // C'est a partir de cette interface qu'on récupère toutes les infos.
  // Il faut donc préciser à l'user qu'il doit avoir totalement rempli
  // le STO (valeurs, nom, path).
  // (nom et path peuvent être définis après coup avec editRefFileEntry).
  virtual void addSimpleTableOutputEntry(ISimpleTableOutput* ptr_sto) = 0;

  virtual void readSimpleTableOutputEntry() = 0;

  // Permet de def nom et path après coup (et de forcer un autre path).
  virtual void editRefFileEntry(String path, String name, bool no_edit_path) = 0;

  // Permet d'écrire le reffile. Modifie auto le path (ce n'est pas à l'user de le faire).
  // (sauf si editRefFileEntry(,,true)).
  virtual bool writeRefFile(Integer only_proc = -1) = 0;
  virtual bool writeRefFile(String path, Integer only_proc = -1) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
