// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGetVariablesValuesParallelOperation.h                      (C) 2000-2006 */
/*                                                                           */
/* Opérations pour accéder aux valeurs de variables d'un autre sous-domaine. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGETVARIABLESVALUESPARALLELOPERATION_H
#define ARCANE_IGETVARIABLESVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations pour accéder aux valeurs de variables d'un autre sous-domaine.
 * \todo utiliser la serialisation+templates pour supporter tout type de variable.
 */
class IGetVariablesValuesParallelOperation
{
 public:
  virtual ~IGetVariablesValuesParallelOperation() {}
 public:
  virtual IParallelMng* parallelMng() =0;
 public:
  /*!
    \brief Récupère les valeurs d'une variable sur des entités distantes

    Cette opération permet de récupérer les valeurs de la variable
    \a variable sur des entités qui ne se trouvent pas dans ce sous-domaine.
    Le tableau \a unique_ids contient le numéro <b>unique</b> des entités
    dont on souhaite récupérer la valeur. Ces valeurs seront stockées
    dans \a values.

    Cette méthode nécessaite en général beaucoup de communications
    car il faut rechercher dans quel sous-domaine appartient les entités
    à partir de leur uniqueId(). Si on connait le sous-domaime, il
    vaut utiliser la méthode surchargé avec ce paramètre.

    \a unique_ids et \a values doivent avoir le même nombre d'éléments.

    Cette opération est collective et bloquante.
   */
  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 RealArrayView values) =0;
  /*!
    \brief Récupère les valeurs d'une variable sur des entités distantes

    Cette opération permet de récupérer les valeurs de la variable
    \a variable sur des entités qui ne se trouvent pas dans ce sous-domaine.
    Le tableau \a unique_ids contient le numéro <b>unique</b> des entités
    dont on souhaite récupérer la valeur et \a sub_domain_ids le sous-domaine
    dans lequel se trouve les entités. Ces valeurs seront stockées dans \a values.

    \a unique_ids, \a sub_domain_ids et \a values doivent avoir
    le même nombre d'éléments.

    Cette opération est collective et bloquante.
   */
  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 Int32ConstArrayView sub_domain_ids,
                                 RealArrayView values) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
