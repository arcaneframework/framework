// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryTransformer.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface d'un objet transformant les courbes d'historiques.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYTRANSFORMER_H
#define ARCANE_CORE_ITIMEHISTORYTRANSFORMER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un objet transformant les courbes d'historiques.
 *
 * Les classes implémentant cette interface peuvent transformer les
 * courbes des historiques en temps. Cela permet par exemple de modifier
 * les points des courbes ou d'en enlever. 
 *
 * L'utilisation se fait via l'appel à ITimeHistoryMng::applyTransformation() qui
 * va appeler La méthode transform() de cette instance pour chaque courbe.
 *
 * Il est permis de changer le nombre d'éléments de la courbe, mais
 * infos.iterations.size()*infos.sub_size et values.size() doivent avoir le même nombre
 * d'éléments. Il n'est pas permis de changer le nom de la courbe ni
 * le nombre de valeurs par itération (sub_size).
 */
class ARCANE_CORE_EXPORT ITimeHistoryTransformer
{
 public:

  //! Infos communes à chaque courbe
  class CommonInfo
  {
   public:

    //! Nom de la courbe
    String name;
    //! Liste des itérations
    Int32SharedArray iterations;
    //! Nombre de valeurs par courbe
    Integer sub_size = 0;
  };

 public:

  virtual ~ITimeHistoryTransformer() = default; //!< Libère les ressources

 public:

  //! Applique la transformation pour une courbe avec des valeurs \a Real
  virtual void transform(CommonInfo& infos, RealSharedArray values) = 0;

  //! Applique la transformation pour une courbe avec des valeurs \a Int32
  virtual void transform(CommonInfo& infos, Int32SharedArray values) = 0;

  //! Applique la transformation pour une courbe avec des valeurs \a Int64
  virtual void transform(CommonInfo& infos, Int64SharedArray values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
