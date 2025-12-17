// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TKiss.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Ce fichier définit le patron de classe TKiss ainsi que la classe associée */
/* Kiss.                                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_TKISS_H
#define ARCANE_CORE_RANDOM_TKISS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/RandomGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::random
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! Patron de classe Kiss. Il permet de définir des classes de générateurs
 * de type Kiss.  Les nombres pseudo-aléatoires générés sont de type UIntType.
 * La génération de ces nombres s'effectue par l'appel de l'opérateur (). L'état
 * du genérateur est défini par un membre private \c _state[i] de la classe qui 
 * est un tableau de cinq éléments (0<i<=4). La graine (état \c state[i] 0<i<=4 
 * initial du générateur appelé également tableau de graines) est initialisée 
 * par l'appel au constructeur ou les différentes méthodes \c seed existantes.
*/
template<typename UIntType, UIntType val>
class TKiss
{
 public:
  typedef UIntType result_type;
  typedef UIntType  state_type;
  static const bool has_fixed_range = true;
  static const result_type min_value = 0 ;
  static const result_type max_value = 4294967295U;
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur minimum possible d'une séquence.
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  result_type min() const { return  min_value; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur maximum possible d'une séquence.
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  result_type max() const { return max_value; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir des
   *         valeurs des arguments. 
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */  
  explicit TKiss(UIntType x0 = 30903, UIntType y0 = 30903, UIntType z0 = 30903, UIntType w0 = 30903, UIntType carry0 = 0)
  { 
    _state[0] = x0;
    _state[1] = y0;
    _state[2] = z0;
    _state[3] = w0;
    _state[4] = carry0;
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief  Initialisation du tableau de graines à partir de l'état \c state.
   *          L'état du générateur \c state doit être composé de cinq éléments.
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  void seed(UIntType *  state)
  { for (Integer i=0;i<5;i++) _state[i] = state[i];}
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief  Initialisation du tableau de graines à partir de la valeur \c x0.  
   *          Le tableau de graines de ce générateur est composé de cinq éléments. 
   *          Les quatre premiers éléments prennent la valeur \c x0. Le cinquième
   *          élément prend la valeur nulle.
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */ 
  void seed(UIntType  x0)
  { for (Integer i=0;i<4;i++) _state[i] = x0;
    _state[4] = 0;
}
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation du tableau de graines à partir des valeurs des
   *         arguments. 
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */ 
  void seed(UIntType  x0,UIntType  y0,UIntType  z0,UIntType  w0,UIntType  carry0)
  { _state[0] = x0; _state[1] = y0;_state[2] = z0;_state[3] = w0;_state[4] = carry0;}
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Méthode qui retourne la composante i del'état du générateur. L'état
   *         complet du générateur est donnée par les valeurs d'index \c i 
   *         comprises entre 0 et 4 ( 0 < \c i <= 4 ).
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */ 
  UIntType  getState(Integer i) const { return _state[i]; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur \c () qui retourne la valeur pseudo 
   *         aléatoire. L'état du générateur est modifié.  
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */
  UIntType  operator()()
  {
    UIntType t;
    _state[0] = _state[0] * 69069 + 1;
    _state[1] ^= _state[1] << 13;
    _state[1] ^= _state[1] >> 17;
    _state[1] ^= _state[1] << 5;

    t = (_state[3]<<1) + _state[2] + _state[4];
    _state[4] = ((_state[2]>>2) + (_state[3]>>3) + (_state[4]>>2)) >>30;
    _state[2]=_state[3];
    _state[3]=t;
    return (_state[0] + _state[1] + _state[2]);
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Fonction de validation (je ne sais pas trop a quoi elle sert!)
   *
   * \author Patrick Rathouit
   * \date   28/07/2006
   */
  bool validation(UIntType x) const { return val == x; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur ==
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */
  bool operator==(const TKiss& rhs) const
  { return (_state[0]  == rhs._state[0]) && (_state[1] == rhs._state[1]) &&  (_state[2] == rhs._state[2])  &&  (_state[3] == rhs._state[3]) &&  (_state[4] == rhs._state[4]) ; }

 private:

  state_type _state[5];

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef TKiss<UInt32, 0> Kiss;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
