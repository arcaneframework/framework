// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRandomNumberGenerator.h                                    (C) 2000-2022 */
/*                                                                           */
/* Interface pour générateur de nombres aléatoires.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef IRANDOMNUMBERGENERATOR_H
#define IRANDOMNUMBERGENERATOR_H

#include <arcane/ItemTypes.h>

using namespace Arcane;

class IRandomNumberGenerator
{
 public:
  virtual ~IRandomNumberGenerator(){};

 public:
  /**
   * @brief Méthode permettant d'initialiser le service.
   * 
   * Avec la graine en option (ou la graine par défaut si l'on est
   * en mode singleton).
   * 
   */
  virtual void initSeed() = 0;

  /**
   * @brief Méthode permettant d'initialiser le service.
   * 
   * @param seed La graine d'origine.
   */
  virtual void initSeed(Int64 seed) = 0;

  /**
   * @brief Méthode permettant de récupérer la graine actuelle.
   * 
   * @return Int64 La graine.
   */
  virtual Int64 seed() = 0;

  /**
   * @brief Méthode permettant de générer une autre graine à partir de
   * la graine en mémoire.
   * 
   * @return Int64 La nouvelle graine.
   */
  virtual Int64 generateRandomSeed() = 0;

  /**
   * @brief Méthode permettant de générer une autre graine à partir de
   * la graine transmise en paramètre.
   * 
   * Cette méthode n'utilise pas la graine en mémoire.
   * 
   * @param parent_seed La graine d'origine.
   * @return Int64 La nouvelle graine.
   */
  virtual Int64 generateRandomSeed(Int64* parent_seed) = 0;

  /**
   * @brief Méthode permettant de générer un nombre aléatoire avec
   * la graine en mémoire.
   * 
   * @return Real Le nombre généré (entre 0 et 1).
   */
  virtual Real generateRandomNumber() = 0;

  /**
   * @brief Méthode permettant de générer un nombre aléatoire avec
   * la graine transmise en paramètre.
   *    
   * Cette méthode n'utilise pas la graine en mémoire.
   * 
   * @param seed La graine.
   * @return Real Le nombre généré (entre 0 et 1).
   */
  virtual Real generateRandomNumber(Int64* seed) = 0;
};

#endif
