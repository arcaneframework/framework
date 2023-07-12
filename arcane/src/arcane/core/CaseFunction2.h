// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunction2.h                                             (C) 2000-2023 */
/*                                                                           */
/* Fonction du jeu de données avec type de valeur explicite.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEFUNCTION2_H
#define ARCANE_CORE_CASEFUNCTION2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/core/CaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation de CaseFunction permettant de retourner directement
 * la valeur associée à un paramètre sans passer par une référence.
 *
 * Cela est principalement utilisé pour simplifier les extensions C# en évitant
 * les différentes surcharges de value().
 *
 * Pour utiliser cette classe, il faut implémenter les méthodes 'valueAs*'
 * pour les deux types d'argument \a Integer et \a Real et pour les
 * différents types de retour possibles.
 */
class ARCANE_CORE_EXPORT CaseFunction2
: public CaseFunction
{
 public:

  //! Construit une fonction du jeu de données.
  explicit CaseFunction2(const CaseFunctionBuildInfo& cfbi)
  : CaseFunction(cfbi)
  {}

 protected:

  void value(Real param, Real& v) const override
  {
    v = valueAsReal(param);
  }
  void value(Real param, Integer& v) const override
  {
    v = valueAsInteger(param);
  }
  void value(Real param, bool& v) const override
  {
    v = valueAsBool(param);
  }
  void value(Real param, String& v) const override
  {
    v = valueAsString(param);
  }
  void value(Real param, Real3& v) const override
  {
    v = valueAsReal3(param);
  }
  void value(Integer param, Real& v) const override
  {
    v = valueAsReal(param);
  }
  void value(Integer param, Integer& v) const override
  {
    v = valueAsInteger(param);
  }
  void value(Integer param, bool& v) const override
  {
    v = valueAsBool(param);
  }
  void value(Integer param, String& v) const override
  {
    v = valueAsString(param);
  }
  void value(Integer param, Real3& v) const override
  {
    v = valueAsReal3(param);
  }

 public:

  virtual Real valueAsReal(Real param) const = 0;
  virtual Integer valueAsInteger(Real param) const = 0;
  virtual bool valueAsBool(Real param) const = 0;
  virtual String valueAsString(Real param) const = 0;
  virtual Real3 valueAsReal3(Real param) const = 0;

  virtual Real valueAsReal(Integer param) const = 0;
  virtual Integer valueAsInteger(Integer param) const = 0;
  virtual bool valueAsBool(Integer param) const = 0;
  virtual String valueAsString(Integer param) const = 0;
  virtual Real3 valueAsReal3(Integer param) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
