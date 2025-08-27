// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mrg32k3a.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Ce fichier définit le patron de classe TMrg32k3a ainsi que la classe      */
/* associée Mrg32k3a.                                                        */ 
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_MRG32K3A_H
#define ARCANE_CORE_RANDOM_MRG32K3A_H
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

/*! Patron de classe TMrg32k3a. Il permet de définir des classes de 
 * générateurs de type Mrg32k3a. Les nombres pseudo-aléatoires générés sont de  
 * type RealType. L'état du générateur est caractérisé par six valeurs de type 
 * RealType et peut être géré en interne par le membre private _state[i] 0<i<=5.
 *
 * La génération d'une séquence de nombres pseudo-aléatoires s'effectue :
 *
 * - soit par l'appel successif de l'opérateur (). Dans ce cas, la graine peut  
 *   etre initialisée par les différentes méthodes seed ou lors de l'appel au 
 *   constructeur. L'état du générateur est géré en interne par l'intermédiaire
 *   du membre private _state[i] (0<i<=5). Ses composantes i sont accessibles  
 *   via la méthode getState(i).
 *
 * - soit par l'appel de la méthode \c apply(value). L'état du générateur est 
 *   géré à l'extérieur de la classe. Les méthodes \c seed et \c getState n'ont 
 *   pas de sens dans cette utilisation.
*/
template<typename RealType, Int32 val>
class TMrg32k3a
{
 public:
  typedef RealType result_type;
  typedef RealType state_type;
  static const bool has_fixed_range = true;
  static const Int32 min_value=0;
  static const Int32 max_value=1;
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir de
   *         la valeur \c x0. L'appel à la méthode \c seed(x0) est réalisé.
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */  
  explicit TMrg32k3a(Int32 x0 = 1)
  {
    seed(x0);
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation du tableau de graines à partir du
   *         tableau \c state. \c state doit être un tableau de six éléments. 
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */  
  explicit TMrg32k3a(state_type *state)
  {
    for(Integer  i=0;i<6;i++)
    _state[i] = state[i];
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief  Initialisation du tableau de graines à partir de la valeur \c x0. 
   *          Le tableau de graines de ce générateur est composé de six éléments. 
   *
   * \author  Patrick Rathouit 
   * \date    28/07/2006
   */
  void seed(Int32 x0) { 
    x0 = (x0 | 1);
    _state[0] = (state_type) x0;
    _state[1] = _state[0];
    _state[2] = _state[1];
    _state[3] = _state[2];
    _state[4] = _state[3];
    _state[5] = _state[4]; 
}
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Méthode qui retourne l'état du générateur pour l'index \c i. L'état
   *         complet du générateur est donnée par les valeurs d'index \c i comprises
   *         entre 0 et 5 ( 0 < \c i <=5 ).
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */ 
  RealType  getState(Integer i) const { return _state[i]; }
 /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur \c () qui retourne la valeur pseudo 
   *         aléatoire du générateur. L'état du générateur est modifié. 
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */
  RealType operator()()
  {
    RealType _x;
    _x = apply(_state);
    return _x;
  }
 /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur pseudo aléatoire à partir de l'état \c state.
   *         L'état du générateur state doit être composé de six éléments.
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  static RealType apply(state_type* state)
  {
    long k;
    Real p;
    p = 1403580.0*state[1] - 810728.0*state[0];
    k = static_cast<long>(p/4294967087.0); p-=k*4294967087.0; if (p <0.0) p+=4294967087.0;
    state[0] = state[1]; state[1]=state[2]; state[2]=p;

    p=527612.0*state[5] - 1370589.0*state[3];
    k=static_cast<long>(p/4294944443.0); p-= k*4294944443.0; if(p<0.0) p+=4294944443.0;
    state[3] = state[4]; state[4]=state[5]; state[5]=p;

    if(state[2] <= state[5]) return ((state[2]-state[5]+4294967087.0)/4294967087.0);
    else return ((state[2]-state[5]) / 4294967087.0);
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur minimum possible d'une séquence. 
   *
   * \author Patrick Rathouit 
   * \date 28/07/2006
   */
  result_type min() const  { return static_cast<result_type>(min_value); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur maximum possible d'une séquence. 
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */
  result_type max() const { return static_cast<result_type>(max_value); }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Fonction de validation (je ne sais pas trop a quoi elle sert!)
   *
   * \author Patrick Rathouit
   * \date 28/07/2006
   */
  bool validation(RealType x) const { return val == x; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur ==
   *
   * \author Patrick Rathouit 
   * \date   28/07/2006
   */
  bool operator==(const TMrg32k3a& rhs) const
  { return (_state[0]  == rhs._state[0]) && (_state[1] == rhs._state[1]) &&  (_state[2] == rhs._state[2])  &&  (_state[3] == rhs._state[3]) &&  (_state[4] == rhs._state[4]) &&  (_state[5] == rhs._state[5]) ; }

 private:

  state_type _state[6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef TMrg32k3a<Real,0> Mrg32k3a;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
