// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryCurveWriter.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface d'un écrivain d'une courbe d'un historique.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYCURVEWRITER_H
#define ARCANE_CORE_ITIMEHISTORYCURVEWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un écrivain d'une courbe.
 *
 * \deprecated Utiliser l'interface ITimeHistoryCurveWriter2 à la place.
 */
class ITimeHistoryCurveWriter
{
 public:

  virtual ~ITimeHistoryCurveWriter() = default; //!< Libère les ressources

 public:

  virtual void build() = 0;

  /*!
   * \brief Écrit la courbe de nom \a name.
   *
   * Les valeurs sont dans le tableau \a values. \a times et \a iterations
   * contiennent respectivement le temps et le numéro de l'itération pour
   * chaque valeur.
   * \a path contient le répertoire où seront écrites les courbes
   */
  virtual void writeCurve(const IDirectory& path,
                          const String& name,
                          Int32ConstArrayView iterations,
                          RealConstArrayView times,
                          RealConstArrayView values,
                          Integer sub_size) = 0;

  //! Nom de l'écrivain
  virtual String name() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
